import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
import matplotlib.pyplot as plt
import numpy as np


# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################
########### 변수 설정 #########################
#############################################

# SPSA 변수
EPSILON = 0.001
ALPHA = 0.602
GAMMA = 0.101

# DeltaLearner 변수
DELTALEARNER_EPSILON = 0.1

whisper_version = "openai/whisper-small"
# whisper_version = "openai/whisper-large-v3"

if whisper_version == "openai/whisper-small":
    NUM_MEL_BINS = 80
elif whisper_version == "openai/whisper-large-v3":
    NUM_MEL_BINS = 128
else:
    raise "위스퍼 버전 확인 필요"

LOSS_FN = "wer" # or ""

#############################################
#############################################

# DeltaLearner 정의
class DeltaLearner(nn.Module):
    def __init__(self, num_mel_bins=128, max_frames=3000, eps=DELTALEARNER_EPSILON):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(1, num_mel_bins, max_frames))  # learnable delta
        self.eps = eps

    def forward(self, mel):
        """
        mel: (batch_size, num_mel_bins, frames)
        """
        batch_size, num_mel_bins, frames = mel.shape

        # Adjust delta size if necessary
        if frames > self.delta.size(2):
            raise ValueError("Input frames exceed max_frames")
        delta_resized = self.delta[:, :, :frames]  # Crop delta to input size
        delta_expanded = delta_resized.expand(batch_size, -1, -1)  # Repeat delta for batch

        return mel + self.eps * delta_expanded  # Add delta to mel
    

# Loss 계산 함수 정의
def calculate_loss(whisper_model, mel_with_delta, labels):
    """
    mel_with_delta: log_mel_spectrogram + delta
    labels: ground truth token IDs
    """
    outputs = whisper_model(input_features=mel_with_delta, labels=labels)

    loss = outputs.loss  # CrossEntropy loss
    return loss

import evaluate

wer_metric = evaluate.load("wer")

def calculate_wer(predictions, references):
    return wer_metric.compute(predictions=predictions, references=references)


# SPSA 업데이트 함수 정의
def spsa_update(delta_learner, whisper_model, mel, labels, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, step=1, loss=LOSS_FN):
    """
    delta_learner: DeltaLearner instance
    whisper_model: Whisper model (frozen)
    mel: log_mel_spectrogram
    labels: ground truth token IDs
    """
    delta_params = delta_learner.delta

    # 랜덤 perturbation 생성 (Rademacher 분포 사용)
    perturb = torch.sign(torch.randn_like(delta_params)) * epsilon

    if loss == "wer":
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Positive perturbation
        delta_learner.delta.data = delta_params + perturb
        mel_with_delta = delta_learner(mel)  # mel + delta
        predictions = whisper_model.generate(input_features=mel_with_delta)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        loss_plus = calculate_wer(pred_texts, ref_texts)

        # Negative perturbation
        delta_learner.delta.data = delta_params - perturb
        mel_with_delta = delta_learner(mel)  # mel + delta
        predictions = whisper_model.generate(input_features=mel_with_delta)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        loss_minus = calculate_wer(pred_texts, ref_texts)

        # SPSA gradient 근사
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon * perturb)

        # 학습 스텝에 따른 학습률 계산
        learning_rate = alpha / ((step + 1) ** gamma)

        # Delta 업데이트
        delta_learner.delta.data -= learning_rate * grad_estimate

        return loss_plus, loss_minus, grad_estimate

    else:
        # Positive perturbation
        delta_learner.delta.data = delta_params + perturb
        loss_plus = calculate_loss(whisper_model, delta_learner(mel), labels)

        # Negative perturbation
        delta_learner.delta.data = delta_params - perturb
        loss_minus = calculate_loss(whisper_model, delta_learner(mel), labels)

        # SPSA gradient 근사
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon * perturb)

        # 학습 스텝에 따른 학습률 계산
        learning_rate = alpha / ((step + 1) ** gamma)

        # Delta 업데이트
        delta_learner.delta.data -= learning_rate * grad_estimate

        return loss_plus, loss_minus, grad_estimate


# 학습 데이터 준비
from datasets import load_dataset

DATASET_ID = "Jzuluaga/atcosim_corpus"
dataset = load_dataset(DATASET_ID, "default", split="train[:1%]")  # 1% 데이터만 사용


processor = WhisperProcessor.from_pretrained(whisper_version)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_version).to(device)
tokenizer = WhisperTokenizer.from_pretrained(whisper_version, language="en", task="transcribe")

# 모델 파라미터 동결
for param in whisper_model.parameters():
    param.requires_grad = False

# 데이터 전처리
def preprocess_data(batch):
    audio = batch["audio"]
    mel = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    labels = processor.tokenizer(batch["text"], return_tensors="pt", padding="longest").input_ids.squeeze(0)

    # mel spectogram 그리기

    # plt.figure(figsize=(10, 4))
    # plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Mel Spectrogram")
    # plt.xlabel("Time Frames")
    # plt.ylabel("Mel Frequency Bins")
    # plt.show()

    return {"mel": torch.tensor(mel, dtype=torch.float32), "labels": labels}

processed_dataset = [preprocess_data(item) for item in dataset]

# 데이터셋 샘플 확인
print(f"Processed dataset sample: mel shape = {processed_dataset[0]['mel'].shape}, labels shape = {processed_dataset[0]['labels'].shape}")

# DeltaLearner 초기화
delta_learner = DeltaLearner(num_mel_bins=NUM_MEL_BINS, max_frames=3000, eps=0.000).to(device) 



# 학습 루프
max_steps = 1  # 적당히 줄인 학습 스텝
log_interval = 1

# 학습 루프 수정
for step, batch in enumerate(processed_dataset):
    # 데이터 준비
    mel = batch["mel"].unsqueeze(0).to(device)  # Add batch dimension
    labels = batch["labels"].unsqueeze(0).to(device)  # Add batch dimension

    # SPSA 업데이트
    spsa_update(delta_learner, whisper_model, mel, labels, step=step, loss=LOSS_FN)


    # 학습 진행 상황 로그 출력
    if step % log_interval == 0:
        loss = calculate_loss(whisper_model, delta_learner(mel), labels)
        print(f"Step {step}: Loss = {loss.item()}")

        # WER 계산
        with torch.no_grad():  # 평가 단계에서는 gradient 계산 필요 없음
            mel_with_delta = delta_learner(mel)  # mel + delta
            # print(f"mel_with_delta shape: {mel_with_delta.shape}")
            predictions = whisper_model.generate(input_features=mel_with_delta)
            # print(predictions)
            pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            wer = calculate_wer(pred_texts, ref_texts)
            print(ref_texts[0])
            print(pred_texts[0])
        
        # 로그 출력
        print(f"Step {step}: WER = {wer}")


    if step >= max_steps:
        break

# 학습 완료 후 Delta 저장
torch.save(delta_learner.state_dict(), "delta_learner.pth")
print("Delta learner saved!")