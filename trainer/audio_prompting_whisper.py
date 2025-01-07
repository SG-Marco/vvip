# VVIP/trainers/audio_prompting_whisper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from audio_prompters import AudioPrompter

class AudioPromptingWhisper(nn.Module):
    """
    WhisperForConditionalGeneration을 그대로 감싸서,
    forward()에서 'input_features' 위에 audio_prompter로부터 나온 delta를 더해 준다.
    """
    def __init__(self, base_model_name="openai/whisper-large-v2", max_frames=3000, freeze_whisper=True):
        super().__init__()
        # 1) Whisper 모델 불러오기
        self.whisper = WhisperForConditionalGeneration.from_pretrained(base_model_name)

        # Whisper 사전학습 파라미터 동결 (옵션)
        if freeze_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False

        # 2) AudioPrompter 초기화
        self.prompter = AudioPrompter(num_mel_bins=80, max_frames=max_frames)

    def forward(self, input_features, labels=None, **kwargs):
        """
        - 🤗Transformers Trainer는 model(**batch) 형태로 호출.
        - batch 딕셔너리에는 "input_features", "labels" 등이 들어있음.
        - input_features shape: (batch_size, 80, frames)
        """
        # 오디오 프롬프트 적용
        prompted_features = self.prompter(input_features)  # (B,80,frames)

        # Whisper forward
        result = self.whisper(
            input_features=prompted_features,
            labels=labels,
            **kwargs
        )
        return result
    
    def save_delta(self, path="audio_delta.pth"):
        """
        AudioPrompter에 학습된 delta 파라미터만 저장
        """
        torch.save(self.prompter.state_dict(), path)
        print(f"[AudioPromptingWhisper] Saved delta params to {path}")

    def load_delta(self, path="audio_delta.pth"):
        """
        저장된 delta 파라미터 로드
        """
        self.prompter.load_state_dict(torch.load(path))
        print(f"[AudioPromptingWhisper] Loaded delta params from {path}")

    def gradient_checkpointing_enable(self, **kwargs):
        """
        중간에 self.whisper가 PreTrainedModel이면 
        whisper.gradient_checkpointing_enable() 호출을 중계
        """
        if hasattr(self.whisper, "gradient_checkpointing_enable"):
            self.whisper.gradient_checkpointing_enable(**kwargs)
        else:
            raise AttributeError("Whisper model does not support gradient_checkpointing_enable.")

    def gradient_checkpointing_disable(self):
        if hasattr(self.whisper, "gradient_checkpointing_disable"):
            self.whisper.gradient_checkpointing_disable()
        else:
            raise AttributeError("Whisper model does not support gradient_checkpointing_disable.")