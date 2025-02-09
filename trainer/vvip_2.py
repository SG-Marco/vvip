import torch
import torch.nn as nn
import torch.nn.utils as utils
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, ViTModel
import evaluate

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################
########### 변수 설정 #########################
#############################################

MAX_STEPS = 50
LOG_INTERVER = 1
BATCH_SIZE = 5  # 한 번에 4개의 음성 파일을 처리하도록 설정

# EPSILON = 0.0001 # SPSA Perturbation 크기
# ALPHA = 0.602 # SPSA Learning rate scaling 0.602
# GAMMA = 0.101  # SPSA Decay 0.101
# AK=0.0004 
# CK=0.000025 # gradient 추정시 사용 값
# O=10
# P_TRIGGER_EPSILON = 0.00000005 # p_trigger 업데이트시 사용

EPSILON = 0.001  # 기존보다 10배 증가
ALPHA = 0.602
GAMMA = 0.101
AK = 0.00001  # 기존보다 25배 증가
CK = 0.005  # 기존보다 5배 증가
O = 7  # 기존보다 감소
P_TRIGGER_EPSILON = 0.0000001  # 기존보다 10배 증가

MAX_GRAD = 3000 # Gradient clipping 제한치

'''파라미터	의미	일반적인 값
α (alpha)	학습률 스케일링 계수	0.602 (기본)
γ (gamma)	c_k 감소율 스케일링 계수	0.101 (기본)
ε (epsilon)	Perturbation 크기	0.001 ~ 0.01
a_k (learning rate)	업데이트 크기	1.0 ~ 0.1
c_k (perturbation size)	Perturbation 스케일링 값	0.1 ~ 0.01
k (sp_avg)	Gradient 평균 계산 시 반복 수	1 ~ 10
max_steps	최대 업데이트 스텝 수	100 ~ 10,000
'''

# LOSS_FN = "wer"  # WER 기반 Loss
LOSS_FN = "cross entropy"  # Loss 유형
MAX_FRAMES = 3000
COORDINATOR_HIDDEN_DIM = 768  # ViT hidden_dim
MAX_NEW_TOKENS = 444  # Whisper default Max Tokens 448 - 4. 4: decoder_input_ids 개수
ENCODER_NAME = "google/vit-base-patch16-224-in21k"

# whisper_version = "openai/whisper-large-v3"
whisper_version = "openai/whisper-small"

if whisper_version == "openai/whisper-small":
    NUM_MEL_BINS = 80
elif whisper_version == "openai/whisper-large-v3":
    NUM_MEL_BINS = 128
else:
    raise ValueError("위스퍼 버전 확인 필요")

#############################################
#############################################

########################
##### Coordinator ######
########################

'''
ViT는 3채널 이미지를 입력으로 받음. 그리고 16x16 사이즈 패치롤 처리하기위해 224x224 사이즈를 기대함.
Mel spectogram 은 1채널이고 80x3000(128x3000)이므로 이 문제를 해결해야함

✅ 최적의 해결 방법: Conv2d 변환 + Conv2d 복원

📌 입력 변환 (1채널 → 3채널)
	•	Conv2d(in_channels=1, out_channels=3, kernel_size=1)를 사용하여 변환.
	•	학습 가능한 변환이므로 최적화 가능.

📌 출력 복원 (3채널 → 1채널)
	•	Conv2d(in_channels=3, out_channels=1, kernel_size=1)를 사용하여 다시 1채널로 변환.
	•	Mel Spectrogram 차원(1, mel_bins, frames)을 유지.

✅ 이 방식의 장점:
✔ ViT 입력과 Whisper 출력을 모두 만족
✔ 학습 가능한 변환이므로 성능 최적화 가능
✔ Mel Spectrogram의 정보 손실 최소화

✅ 해결 방법: Mel Spectrogram을 ViT가 처리할 수 있도록 변환

🚀 방법 1: 패치 크기 조정 (patch_size=16 → 커스텀 크기 사용)
	•	ViT는 기본적으로 16×16 패치 단위로 이미지를 분할하여 처리.
	•	하지만 Mel Spectrogram은 길이가 길고 높이가 작음 → 일반적인 16×16 패치 방식이 적합하지 않음.
	•	커스텀 패치 크기를 설정하여 80×3000 데이터를 ViT가 처리 가능하도록 변환.

📌 방법:
	•	ViTFeatureExtractor의 패치 크기를 조정 (patch_size=(8, 64))
	•	Mel Spectrogram을 80×3000 → 224×224로 리사이즈 후 ViT에 입력
'''
import torch
import torch.nn as nn
from transformers import ViTModel


class Coordinator(nn.Module):
    def __init__(self, encoder_name="google/vit-base-patch16-224-in21k", mel_bins=80, frames=3000, src_dim=1568, hidden_dim=768):
        super(Coordinator, self).__init__()       
        self.backbone = encoder_name
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        z_dim = hidden_dim

        self.encoder = ViTModel.from_pretrained(encoder_name)

        # ✅ **Mel Spectrogram 변환 (1채널 → 3채널)**
        self.conv1x1_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        # ✅ **Mel Spectrogram 크기 변환 (80×3000 → 224×224)**
        self.resize = nn.AdaptiveAvgPool2d((224, 224))

        self.dec = DecoderManual(z_dim, src_dim, act=act, arch=self.backbone)

    def forward(self, x):
        with torch.no_grad():
 
            # ✅ **1채널 추가 (batch_size, mel_bins, frames) → (batch_size, 1, mel_bins, frames)**
            if x.dim() == 3:  
                x = x.unsqueeze(1)  # (batch_size, 1, mel_bins, frames)

            # ✅ **1채널 → 3채널 변환**
            x = self.conv1x1_in(x)  # (batch_size, 3, mel_bins, frames)

            # ✅ **80×3000 → 224×224 크기로 변환**
            x = self.resize(x)  # (batch_size, 3, 224, 224)

            #! (N, 197, 768) => pick [CLS] => (N, 768)
            out = self.encoder(x)
            z = out.last_hidden_state[:,0,:]
      
        wrap = self.dec(z)
        return z, wrap

class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManual, self).__init__()
        if i_dim:
            self.shared_feature = 1
        else:     self.shared_feature = 0
        if self.shared_feature:
            #! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1) # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim
        
        bias_flag = False
        body_seq = []
        
        if arch in ['vit-mae-base', 'vit-base', "google/vit-base-patch16-224-in21k"]:
            if src_c >= 64:    g_c = 64
            else:              g_c = src_c
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]  
   
        else: 
            raise ValueError('not implemented')
        self.body   = nn.Sequential(*body_seq)

        # ✅ **3채널 → 1채널 변환 (출력)**
        self.conv1x1_out = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        # ✅ **Mel Spectrogram 크기 변환 (80×3000 → 224×224)**
        self.resize = nn.AdaptiveAvgPool2d((NUM_MEL_BINS, MAX_FRAMES))

    def forward(self, z):
        if self.shared_feature:
            # z.shape = (1, 768)
            N = z.shape[0]
            D = self.p_trigger.shape[1] 
            
            p_trigger = self.p_trigger.repeat(N, 1) # (1, 800)

            z_cube = torch.cat((z, p_trigger), dim=1) # (1, 1568)

            z_cube = z_cube.reshape(N, -1, 7, 7) # (1, 32, 7, 7)

            z_cube = self.body(z_cube) # (1, 3, 224, 224)
     
            # ✅ **224x224 → 80x3000 크기로 변환**
            z_cube = self.resize(z_cube)  # (batch_size, 3, mel_bins, frames)
     
            # ✅ **3채널 → 1채널 변환**
            z_cube = self.conv1x1_out(z_cube)  # (batch_size, 1, mel_bins, frames)
  
            # ✅ **1채널 제거 (batch_size, mel_bins, frames)**
            z_cube = z_cube.squeeze(1)  # (batch_size, mel_bins, frames)
        else:
            return self.body(z)
        return z_cube


############################

# Coordinator 초기화
coordinator = Coordinator(encoder_name=ENCODER_NAME, mel_bins=NUM_MEL_BINS, frames=MAX_FRAMES, hidden_dim=COORDINATOR_HIDDEN_DIM).to(device)


############################
###### 업데이트 로직 설정 ######
############################

from whisper.normalizers import EnglishTextNormalizer

wer_metric = evaluate.load("wer")
normalizer = EnglishTextNormalizer() # normalizer 적용

# def calculate_wer(references, predictions): 
#     return wer_metric.compute(references=references, predictions=predictions)

def calculate_wer(references, predictions, tokenizer):
    """
    패딩된 labels를 무시하고 WER을 계산하는 함수
    """
    with torch.no_grad():  # Gradient 저장 방지
        filtered_references = []
        
        # 패딩을 제외한 원본 labels 추출 & list of characters → list of words 변환
        for ref in references:
            ref_filtered = [word for word in ref if word != tokenizer.pad_token_id]
            filtered_references.append("".join(ref_filtered))  # 🔹 join()을 사용해 문자 리스트를 문자열로 변환

        # ✅ 리스트의 각 요소에 `normalizer()` 적용
        filtered_references = ["".join(ref_filtered) for ref_filtered in filtered_references]  # 리스트 -> 문자열 변환
        filtered_references = [normalizer(ref) for ref in filtered_references]  # 정상화 적용
        predictions = [normalizer(pred) for pred in predictions]  # 정상화 적용

    # ✅ 정상화된 데이터를 WER 계산에 사용
    return wer_metric.compute(references=filtered_references, predictions=predictions)


import torch.nn.functional as F

def calculate_cross_entropy_loss(whisper_model, mel_with_delta, labels):
    """
    mel_with_delta: Coordinator와 결합된 Mel Spectrogram
    labels: ground truth token IDs
    """
    with torch.no_grad():  # Gradient 저장 방지
        outputs = whisper_model(input_features=mel_with_delta, labels=labels)
    return outputs.loss  # CrossEntropy loss

# SPSA 업데이트 함수 (Trigger Vector + Decoder 학습)
# 디코더 전체를 벡터화하여 perturbation 후, 다시 벡터에서 파라미터로 변환
'''
✅ 장점
	1.	SPSA 최적화의 안정성 증가
	•	모든 디코더 파라미터를 단일 벡터 형태로 변환하여 업데이트하므로, 개별 레이어의 변화량을 고려한 전역적인 perturbation 적용이 가능합니다.
	•	이는 개별 레이어에 독립적으로 perturbation을 적용하는 것보다 일관된 변화 패턴을 유지하면서 최적화가 이루어지도록 합니다.
	2.	Gradient Estimation이 효율적으로 이루어짐
	•	디코더 전체를 하나의 벡터로 취급하여 SPSA Gradient를 근사하므로, 개별 파라미터마다 따로 계산하는 것보다 연산량이 줄어들고 학습이 효율적입니다.
	•	특히, 높은 차원의 파라미터를 가진 모델에서는 Gradient 추정이 더 정확해질 수 있음.
	3.	코드의 가독성과 유지보수 용이
	•	parameters_to_vector()를 사용하면 디코더 전체를 하나의 벡터로 취급할 수 있어, 학습 코드가 간결해지고 유지보수가 쉬워집니다.
	•	vector_to_parameters()를 사용하면 perturbation 후 원래 형태의 모델 파라미터로 다시 복구할 수 있어 구조를 유지한 채 업데이트 가능.
	4.	Batch-level 업데이트가 가능
	•	개별 레이어마다 perturbation을 적용하는 방식과 달리, 벡터화된 perturbation을 적용하면 batch-level에서 최적화가 가능.
	•	이는 병렬 연산 및 GPU 활용을 극대화할 수 있는 구조가 되어 학습 속도가 향상될 수 있음.
❌ 단점
	1.	모델이 커질수록 메모리 부담 증가
	•	모든 파라미터를 하나의 벡터로 변환하여 perturbation을 적용하면, 큰 모델에서는 메모리 사용량이 급격히 증가할 수 있음.
	•	특히, 대규모 Transformer 기반 디코더(예: GPT, BERT)에서는 파라미터 개수가 수억 개에 이르기 때문에 벡터화된 방식이 메모리 부족 문제를 일으킬 수 있음.
	2.	SPSA 방식이 일반적인 Gradient Descent보다 학습 속도가 느릴 수 있음
	•	SPSA 자체가 Gradient를 직접 계산하는 것이 아니라, Perturbation을 이용하여 근사하는 방식이므로, 일반적인 SGD나 Adam보다 업데이트의 안정성이 낮아질 가능성이 있음.
	•	따라서 학습 초반에 수렴 속도가 느릴 수 있음.
	3.	Fine-tuning에 적합하지 않을 수 있음
	•	디코더 전체를 벡터화하고 perturbation을 적용하는 방식은 세밀한 조정이 필요한 Fine-tuning 단계에서는 불리할 수 있음.
	•	특정 Layer만 미세 조정하려면, Layer별로 perturbation을 다르게 적용해야 하는데, 벡터화된 방식에서는 이를 수행하기 어려움.
	4.	SPSA 자체의 한계
	•	SPSA는 샘플링 기반 Gradient 근사 방식이므로, 정확한 Gradient Descent보다 성능이 떨어질 수 있음.
	•	특히, Loss Surface가 매우 복잡한 경우, SPSA가 지역 최소점(Local Minima)에 빠질 가능성이 높음.
'''

class SPSA:
    def __init__(self, alpha=0.602, gamma=0.101, epsilon=0.01, ak=0.8, ck=0.05, o=0.01, p_trigger_epsilon=0.01):
        super(SPSA, self).__init__()   
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.ak = ak
        self.ck = ck
        self.o = o
        self.p_trigger_epsilon = p_trigger_epsilon


    def parameter_update(self, step):
        self.ak = self.ak / ((step + self.o)**self.alpha)
        self.ck = self.ck / (step**self.gamma)


    def spsa_update(self, coordinator, whisper_model, mel: list, labels: list):
        """
        coordinator: Coordinator instance
        whisper_model: Whisper 모델 (frozen)
        mel: 원본 Mel Spectrogram (batch_size, mel_bins, frames)
        labels: Ground Truth 라벨
        """
        torch.cuda.empty_cache()  # 메모리 캐시 정리

        # **1. 디코더 전체의 파라미터를 벡터화**
        coordinator_params = torch.nn.utils.parameters_to_vector(coordinator.dec.parameters()).detach()
        
        # **2. 랜덤 Perturbation 생성**
        perturb = torch.sign(torch.randn_like(coordinator_params)) * self.epsilon

        # **3. Positive Perturbation 적용**
        perturbed_params = coordinator_params + perturb
        torch.nn.utils.vector_to_parameters(perturbed_params, coordinator.dec.parameters())

        _, mel_transformed = coordinator(mel)  # 전체 배치 변환
        mel_with_delta = mel + mel_transformed  
        if LOSS_FN == "wer":
            # Whisper 모델을 배치 단위로 호출
            predictions = whisper_model.generate(input_features=mel_with_delta, max_new_tokens=MAX_NEW_TOKENS, language="en")

            # 배치 단위 WER 계산
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            loss_plus = calculate_wer(ref_texts, pred_texts, tokenizer)  # ✅ 배치 전체 WER 계산

        elif LOSS_FN == "cross entropy":
            # Whisper 모델을 배치 단위로 호출
            loss_plus = calculate_cross_entropy_loss(whisper_model, mel_with_delta, labels)
        else:
            raise ValueError("Loss function not supported")

        # **4. Negative Perturbation 적용**

        perturbed_params = coordinator_params - perturb
        torch.nn.utils.vector_to_parameters(perturbed_params, coordinator.dec.parameters())

        _, mel_transformed = coordinator(mel)  # 전체 배치 변환
        mel_with_delta = mel + mel_transformed  

        # Whisper 모델을 배치 단위로 호출
        predictions = whisper_model.generate(input_features=mel_with_delta, max_new_tokens=MAX_NEW_TOKENS, language="en")

        # 배치 단위 WER 계산
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        print("Ref", ref_texts)
        print("Pred",pred_texts)

        if LOSS_FN == "wer":
            loss_minus = calculate_wer(ref_texts, pred_texts, tokenizer)  # ✅ 배치 전체 WER 계산

        elif LOSS_FN == "cross entropy":
            # Whisper 모델을 배치 단위로 호출
            loss_minus = calculate_cross_entropy_loss(whisper_model, mel_with_delta, labels)
        else:
            raise ValueError("Loss function not supported")

        # 배치 내 모든 샘플의 평균 gradient 계산
        grad_estimate_avg =  (loss_plus - loss_minus) / (2 * self.ck * perturb)

        # Gradient Clipping 적용
        grad_estimate_avg = torch.clamp(grad_estimate_avg, min=-MAX_GRAD, max=MAX_GRAD)

        # coordinator parameters 원래 값으로 다시 변경
        torch.nn.utils.vector_to_parameters(coordinator_params, coordinator.dec.parameters())

        # SPSA Gradient 근사
        print(f"Gradient estimate: {grad_estimate_avg}")

        learning_rate = self.ak

        # **5. Coordinator 업데이트**
        # p_trigger와 decoder의 파라미터 개수 계산
        p_trigger_param_num = coordinator.dec.p_trigger.numel()

        # p_trigger와 decoder의 gradient를 분리
        p_trigger_grad = grad_estimate_avg[:p_trigger_param_num]  # 앞부분 → p_trigger의 gradient
        decoder_grad = grad_estimate_avg[p_trigger_param_num:]  # 뒷부분 → decoder의 gradient

        # **p_trigger 업데이트**
        p_trigger_vector = torch.nn.utils.parameters_to_vector([coordinator.dec.p_trigger])  # 기존 p_trigger 값 벡터화
        p_trigger_update = p_trigger_vector - learning_rate * p_trigger_grad * self.p_trigger_epsilon  # 기존 값에서 업데이트 계산
        torch.nn.utils.vector_to_parameters(p_trigger_update, [coordinator.dec.p_trigger])  # 업데이트 적용

        # **decoder 업데이트**
        decoder_params = [p for name, p in coordinator.dec.named_parameters() if name != "p_trigger"]
        decoder_vector = torch.nn.utils.parameters_to_vector(decoder_params)  # 기존 decoder 값 벡터화
        decoder_update = decoder_vector - learning_rate * decoder_grad  # 기존 값에서 업데이트 계산
        torch.nn.utils.vector_to_parameters(decoder_update, decoder_params)  # 업데이트 적용

        return loss_plus, loss_minus, grad_estimate_avg


#########################################
####### Whisper 모델 로드 및 Freeze ########
#########################################

processor = WhisperProcessor.from_pretrained(whisper_version)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_version).to(device)
tokenizer = WhisperTokenizer.from_pretrained(whisper_version, language="en", task="transcribe")

 # Whisper 모델 Freeze
for param in whisper_model.parameters():
    param.requires_grad = False 



#######################################
######### 데이터 준비 및 모델 초기화 #########
#######################################

from datasets import load_dataset

DATASET_ID = "Jzuluaga/atcosim_corpus"
dataset = load_dataset(DATASET_ID, "default", split="train[:2%]")  # 데이터 일부만 사용

# 데이터 전처리
def preprocess_data(batch):
    audio = batch["audio"]
    mel = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    labels = processor.tokenizer(batch["text"], return_tensors="pt", padding="longest").input_ids.squeeze(0)
    return {"mel": torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device), "labels": labels.to(device)}

    
processed_dataset = [preprocess_data(item) for item in dataset]

# ✅ CustomDataset 및 DataLoader 정의
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

# ✅ collate_fn 정의 → 다른 길이의 labels 처리
def collate_fn(batch):
    mel_batch = torch.stack([item["mel"].squeeze(0) for item in batch])  # Mel-Spectrogram 배치화
    labels_batch = [item["labels"] for item in batch]  # Labels 리스트로 유지

    # ✅ 가장 긴 labels에 맞게 패딩
    labels_padded = pad_sequence(labels_batch, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {"mel": mel_batch, "labels": labels_padded}

# ✅ DataLoader 적용 (배치 크기 지정)
dataset = CustomDataset(processed_dataset)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  # ✅ collate_fn 추가

###################
##### 학습 루프 #####
###################

print("Update method: ", LOSS_FN)
spsa = SPSA(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, ak=AK, ck=CK, o=O, p_trigger_epsilon=P_TRIGGER_EPSILON)
avg_losses = []

for epoch in range(MAX_STEPS):  # MAX_STEPS 만큼 반복
    total_loss = 0.0 
    num_batches = 0  # 배치 개수 카운트

    for batch in dataloader:
        mel = batch["mel"].to(device)  # 배치 데이터를 텐서로 변환
        labels = batch["labels"].to(device)  # 패딩된 labels 텐서 변환

        loss_plus, loss_minus, grad_estimate_avg = spsa.spsa_update(coordinator, whisper_model, mel, labels)

        loss = (loss_plus + loss_minus) / 2
        total_loss += loss
        num_batches += 1

        spsa.parameter_update(epoch+1)

    # 해당 에폭의 평균 Loss 계산
    avg_loss = total_loss / (num_batches * BATCH_SIZE)
    avg_losses.append(avg_loss)

    print()
    print(f"Epoch {epoch}: Avg WER Loss = {avg_loss:.4f}--------------------%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@+++++++++++++++++++++")
    print()

    if epoch >= MAX_STEPS:
        break

# 학습 완료 후 모델 저장
torch.save(coordinator.state_dict(), "coordinator.pth")
print("Coordinator saved!")

import matplotlib.pyplot as plt

# GPU 텐서를 CPU로 옮긴 후 NumPy 배열로 변환
avg_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in avg_losses]

plt.plot(avg_losses_cpu)
plt.xlabel('Epoch')
plt.ylabel(f'Average {LOSS_FN}')
plt.title(f'Average {LOSS_FN} Over Epochs')
plt.show()
