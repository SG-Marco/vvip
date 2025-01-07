# trainers/audio_prompters.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioPrompter(nn.Module):
    """
    간단한 예시:
      - learnable parameter (1, 1, T, F) 형태를 배치에 repeat해서 mel과 더함
      - T, F가 가변이면 runtime에서 shape에 맞춰 interpolate하는 방식을 써야 함
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 예시로 T=3000, F=128처럼 cfg에서 받아올 수 있음
        # 여기서는 하드코딩하거나, None으로 두고 forward에서 처리
        self.T = cfg.TRAINER.BLACKVIP.AUDIO_T  # 예: 3000
        self.F = cfg.TRAINER.BLACKVIP.AUDIO_F  # 예: 128

        # 실제 학습 파라미터 (delta)
        # 보통 초기값은 0으로 두고, SPSA나 Adam으로 업데이트
        self.delta = nn.Parameter(torch.zeros(1, 1, self.T, self.F))

    def forward(self, mel):
        """
        mel: (B, 1, T, F)  Whisper의 log_mel_spectrogram 형태
        return: (delta, None)
          - BlackVIP의 CustomCLIP이 (prompt, _) 튜플을 받는 구조를 맞추기 위해
        """
        B, C, T, F = mel.shape

        # 만약 T, F != self.T, self.F 라면 interpolate 필요
        if T != self.T or F != self.F:
            # bilinear interpolate (2D)
            delta_resized = F.interpolate(
                self.delta,
                size=(T, F),
                mode='bilinear',
                align_corners=False
            )
        else:
            delta_resized = self.delta

        # 배치만큼 반복
        # delta_resized: (1, 1, T, F) -> (B, 1, T, F)
        prompt_delta = delta_resized.repeat(B, 1, 1, 1)

        # BlackVIP 형식에 맞춰 (prompt, None)을 리턴
        return prompt_delta, None


def audio_coordinator(cfg):
    """
    config에서 `METHOD`로 audio_coordinator를 지정하면 여기로 들어와서
    AudioPrompter를 생성한다고 가정
    """
    return AudioPrompter(cfg)

###############################

# audio_prompter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioPrompter(nn.Module):
    """
    아주 간단한 오디오 프롬터:
      - (1, 80, M) 모양의 학습 파라미터(delta)를 배치 크기에 맞춰 repeat한 뒤 mel에 add
      - M(프레임 수) 부분은 실제 입력 음성 길이에 따라 달라질 수 있어,
        필요하면 forward()에서 interpolation 로직을 추가할 수 있음.
    """
    def __init__(self, num_mel_bins=80, max_frames=3000):
        super().__init__()
        # (1, 80, max_frames)짜리 learnable 텐서
        self.delta = nn.Parameter(torch.zeros(1, num_mel_bins, max_frames))
        # 필요시 초기화 방식 변경 가능

        # 만약 max_frames보다 작은 길이가 들어오면
        # 나중에 forward()에서 crop이나 interpolation로 맞출 수도 있음

    def forward(self, mel):
        """
        mel: (batch_size, 80, frames) 형태 가정
        return: (batch_size, 80, frames)에 동일 shape의 delta를 더한 결과
        """
        B, mel_bins, frames = mel.shape
        
        if frames > self.delta.shape[2]:
            raise ValueError(f"입력 frames={frames}이 prompter가 지원하는 최대 길이 "
                             f"{self.delta.shape[2]}을 초과했습니다.")

        # delta를 (B, 80, frames)에 맞춰 잘라내거나 interpolate
        # 여기서는 간단히 [:frames]로 자르는 예시
        delta_cropped = self.delta[:, :, :frames]  # shape (1, 80, frames)

        # batch dimension repeat
        delta_expanded = delta_cropped.repeat(B, 1, 1)  # (B, 80, frames)

        # mel + delta
        mel_prompted = mel + delta_expanded
        return mel_prompted