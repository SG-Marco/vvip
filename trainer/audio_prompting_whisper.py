# audio_prompting_whisper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from audio_prompter import AudioPrompter

class AudioPromptingWhisper(nn.Module):
    """
    WhisperForConditionalGeneration을 그대로 감싸고,
    forward()에서 'input_features' 위에 audio_prompter로부터 나온 delta를 더해 줌.
    """
    def __init__(self, base_model_name="openai/whisper-large-v3", max_frames=3000):
        super().__init__()
        # 1) Whisper 모델 불러오기
        self.whisper = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        # Whisper 사전학습 파라미터 동결
        for param in self.whisper.parameters():
            param.requires_grad = False
        # AudioPrompter는 requires_grad=True

        # 2) AudioPrompter 초기화 (num_mel_bins=80 고정, frames=3000 가정)
        self.prompter = AudioPrompter(num_mel_bins=80, max_frames=max_frames)


    def forward(self, input_features, labels=None, **kwargs):
        """
        - 🤗Transformers Trainer는 model(**batch) 형태로 호출.
        - batch 딕셔너리에는 "input_features", "labels" 등이 들어있음.
        - 여기서 input_features shape: (batch_size, 80, frames)
        """
        # 1) 오디오 프롬프트 추가
        #    Whisper expects shape: (batch, 80, frames)
        #    self.prompter는 (B,80,frames)에 delta를 더해줌
        prompted_features = self.prompter(input_features)

        # 2) Whisper forward
        #    WhisperForConditionalGeneration은
        #    forward(input_features=(B,80,frames), labels=(B, seq_len)) 식
        result = self.whisper(
            input_features=prompted_features,
            labels=labels,
            **kwargs
        )

        return result