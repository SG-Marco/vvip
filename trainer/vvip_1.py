import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from coordinator import Coordinator  # Coordinator 클래스 import
import evaluate

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################
########### 변수 설정 #########################
#############################################

MAX_STEPS = 5
LOG_INTERVER = 1

EPSILON = 0.001  # SPSA Perturbation 크기
ALPHA = 0.0602    # SPSA Learning rate scaling 0.602
GAMMA = 0.101    # SPSA Decay 0.101
# LOSS_FN = "cross entropy"  # Loss 유형
LOSS_FN = "wer"  # Loss 유형
MAX_FRAMES = 3000
COORDINATOR_HIDDEN_DIM = 128
MAX_NEW_TOKENS = 448 # 위스퍼 기본 맥스 토큰값

whisper_version = "openai/whisper-small"
# whisper_version = "openai/whisper-large-v3"

if whisper_version == "openai/whisper-small":
    NUM_MEL_BINS = 80
elif whisper_version == "openai/whisper-large-v3":
    NUM_MEL_BINS = 128
else:
    raise ValueError("위스퍼 버전 확인 필요")

#############################################
#############################################

# Loss 계산 함수 정의
def calculate_loss(whisper_model, mel_with_delta, labels):
    """
    mel_with_delta: Coordinator와 결합된 Mel Spectrogram
    labels: ground truth token IDs
    """
    outputs = whisper_model(input_features=mel_with_delta, labels=labels)
    return outputs.loss  # CrossEntropy loss

wer_metric = evaluate.load("wer")

def calculate_wer(predictions, references): # pred, ref 순서 맞는지 확인 필요
    return wer_metric.compute(predictions=predictions, references=references)

# SPSA 업데이트 함수
def spsa_update(coordinator, whisper_model, mel, labels, epsilon, alpha, gamma, step, loss):
    """
    coordinator: Coordinator instance
    whisper_model: Whisper 모델 (frozen)
    mel: 원본 Mel Spectrogram
    labels: Ground Truth 라벨
    """
    coordinator_params = coordinator.encoder.body[0].weight  # Coordinator 가중치
    
    # 랜덤 Perturbation 생성
    perturb = torch.sign(torch.randn_like(coordinator_params)) * epsilon

    if loss == "wer":
        # Positive Perturbation
        # 인코더에 변화 주는거 맞는지?? blackvip에서 인코더는 고정인데.. 디코더가 맞는거 같은데 인코더도 안될건 없을듯. 나만의 알고리즘..?
        coordinator.encoder.body[0].weight.data += perturb  
        z, mel_transformed = coordinator(mel)

        mel_with_delta = mel + mel_transformed  # 결합
        predictions = whisper_model.generate(input_features=mel_with_delta, max_new_tokens=MAX_NEW_TOKENS)
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        loss_plus = calculate_wer(pred_texts, ref_texts)

        # Negative Perturbation
        coordinator.encoder.body[0].weight.data -= 2 * perturb
        z, mel_transformed = coordinator(mel)
        mel_with_delta = mel + mel_transformed  # 결합
        predictions = whisper_model.generate(input_features=mel_with_delta, max_new_tokens=MAX_NEW_TOKENS)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        loss_minus = calculate_wer(pred_texts, ref_texts)

        # SPSA Gradient 근사
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon * perturb)

        # 학습 스텝에 따른 학습률 계산
        learning_rate = alpha / ((step + 1) ** gamma)

        # Coordinator 업데이트
        coordinator.encoder.body[0].weight.data += learning_rate * grad_estimate

        return loss_plus, loss_minus, grad_estimate
    
    elif LOSS_FN == "cross entropy" :
        # Positive perturbation
        coordinator.encoder.body[0].weight.data += perturb
        loss_plus = calculate_loss(whisper_model, coordinator(mel)[1], labels)

        # Negative perturbation
        coordinator.encoder.body[0].weight.data -= perturb * 2
        loss_minus = calculate_loss(whisper_model, coordinator(mel)[1], labels)

        # SPSA gradient 근사
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon * perturb)

        # 학습 스텝에 따른 학습률 계산
        learning_rate = alpha / ((step + 1) ** gamma)

        # Coordinator 업데이트
        coordinator.encoder.body[0].weight.data += learning_rate * grad_estimate

        return loss_plus, loss_minus, grad_estimate

    else:
        raise ValueError("Set loss function")

# 데이터 준비 및 모델 초기화
from datasets import load_dataset

DATASET_ID = "Jzuluaga/atcosim_corpus"
dataset = load_dataset(DATASET_ID, "default", split="train[:1%]")  # 데이터 일부만 사용

processor = WhisperProcessor.from_pretrained(whisper_version)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_version).to(device)
tokenizer = WhisperTokenizer.from_pretrained(whisper_version, language="en", task="transcribe")

# Whisper 모델 파라미터 동결
for param in whisper_model.parameters():
    param.requires_grad = False

# Coordinator 초기화
coordinator = Coordinator(mel_bins=NUM_MEL_BINS, frames=MAX_FRAMES, out_dim=COORDINATOR_HIDDEN_DIM).to(device)

# 데이터 전처리
def preprocess_data(batch):
    audio = batch["audio"]
    mel = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    labels = processor.tokenizer(batch["text"], return_tensors="pt", padding="longest").input_ids.squeeze(0)
    return {"mel": torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device), "labels": labels.to(device)}
    # return {"mel": torch.tensor(mel, dtype=torch.float32).unsqueeze(0), "labels": labels}
    
processed_dataset = [preprocess_data(item) for item in dataset]

# 학습 루프
print("Update method: ", LOSS_FN)
for step, batch in enumerate(processed_dataset):
    mel = batch["mel"].to(device)
    labels = batch["labels"].unsqueeze(0).to(device)

    # SPSA 업데이트
    loss_plus, loss_minus, grad_estimate = spsa_update(
        coordinator, whisper_model, mel, labels, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, step=step, loss=LOSS_FN
    )

    # 진행 상황 출력
    # print(f"Step {step}: Loss+ = {loss_plus}, Loss- = {loss_minus}")

    if step % LOG_INTERVER == 0:
        cross_entropy_loss = calculate_loss(whisper_model, coordinator(mel)[1], labels)


        # WER 계산
        with torch.no_grad():  # 평가 단계에서는 gradient 계산 필요 없음
            z, mel_transformed = coordinator(mel)
            mel_with_delta = mel + mel_transformed 
 
            predictions = whisper_model.generate(input_features=mel_with_delta, max_new_tokens=MAX_NEW_TOKENS)
            pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            wer = calculate_wer(pred_texts, ref_texts)
            print(ref_texts[0])
            print(pred_texts[0])


        print(f"Step {step}: cross_entropy_loss = {cross_entropy_loss.item()}, wer = {wer}")

    # 학습 종료 조건
    if step >= MAX_STEPS:
        break


# 학습 완료 후 Delta 저장
torch.save(coordinator.state_dict(), "coordinator.pth")
print("coordinator saved!")

# z 는 어떻게 저장??
# 지금 상태는 z를 추가하는 구조가 아님