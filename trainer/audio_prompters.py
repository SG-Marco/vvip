# VVIP/trainers/audio_prompter.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioPrompter(nn.Module):
    """
    아주 간단한 오디오 프롬터:
      - (1, 80, max_frames) 모양의 학습 파라미터(delta)를 배치 크기에 맞춰 repeat한 뒤 mel에 add
      - max_frames보다 입력 frames가 작으면 crop, 크면 에러
      - 필요시 interpolation 등 다른 로직을 붙일 수 있음
    """
    def __init__(self, num_mel_bins=80, max_frames=3000):
        super().__init__()
        # (1, 80, max_frames)짜리 learnable 텐서
        self.delta = nn.Parameter(torch.zeros(1, num_mel_bins, max_frames))
        # 초기값 0. 필요에 따라 nn.init.uniform_ 등 적용 가능

    def forward(self, mel):
        """
        mel: (batch_size, 80, frames) 형태
        return: (batch_size, 80, frames)에 동일 shape의 delta를 더한 결과
        """
        B, mel_bins, frames = mel.shape
        
        if frames > self.delta.shape[2]:
            raise ValueError(
                f"[AudioPrompter] 입력 frames={frames}이 "
                f"prompter가 지원하는 최대 길이 {self.delta.shape[2]} 초과."
            )

        # delta를 (B, 80, frames)에 맞춰 잘라내거나 interpolate
        delta_cropped = self.delta[:, :, :frames]  # (1,80,frames)

        # batch dimension repeat
        delta_expanded = delta_cropped.repeat(B, 1, 1)  # (B,80,frames)

        # mel + delta
        mel_prompted = mel + delta_expanded
        return mel_prompted