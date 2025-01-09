import torch
import torch.nn as nn
from torchvision import transforms


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.TRAINER.VPWB.PROMPT_SIZE  # 패딩 크기
        mel_bins = args.TRAINER.VPWB.MEL_BINS     # Mel Spectrogram 높이 (num_mel_bins)
        frames = args.TRAINER.VPWB.FRAMES         # Mel Spectrogram 너비 (frames)

        # Mel Spectrogram 패딩 구조 생성
        self.pad_up = nn.Parameter(torch.randn([1, 1, pad_size, frames]))
        self.pad_down = nn.Parameter(torch.randn([1, 1, pad_size, frames]))
        self.pad_left = nn.Parameter(torch.randn([1, 1, mel_bins, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 1, mel_bins, pad_size]))

    def forward(self, x):
        """
        x: (batch_size, 1, mel_bins, frames) 형태의 Mel Spectrogram 데이터
        """
        padded = torch.cat([self.pad_left, x, self.pad_right], dim=3)  # 좌우 패딩
        padded = torch.cat([self.pad_up, padded, self.pad_down], dim=2)  # 상하 패딩
        return padded


class PROGRAM(nn.Module):
    def __init__(self, args):
        super(PROGRAM, self).__init__()
        self.mel_bins = args.TRAINER.BAR.MEL_BINS  # Mel Spectrogram의 높이
        self.frames = args.TRAINER.BAR.FRAMES      # Mel Spectrogram의 너비

        # Learnable Parameter로 변환된 패턴 생성
        self.W = nn.Parameter(torch.randn(1, 1, self.mel_bins, self.frames))  # 1채널 유지

    def forward(self, target_data):
        """
        target_data: (batch_size, 1, mel_bins, frames) 형태의 Mel Spectrogram
        """
        return target_data + self.W


class EncoderManual(nn.Module):
    def __init__(self, mel_bins, out_dim, act=nn.GELU, gap=False):
        """
        Mel Spectrogram을 입력으로 처리하는 인코더
        """
        super(EncoderManual, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1채널 입력
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 다운샘플링
            nn.ReLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),  # 마지막 출력
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) if gap else None

    def forward(self, x):
        x = self.body(x)
        if self.gap:
            x = self.gap(x)
        return x


class DecoderManual(nn.Module):
    def __init__(self, in_dim, mel_bins, frames, act=nn.GELU):
        """
        Decoder for reconstructing Mel Spectrogram
        """
        super(DecoderManual, self).__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 1채널 출력
        )

    def forward(self, x):
        x = self.body(x)
        return x


class Coordinator(nn.Module):
    def __init__(self, mel_bins, frames, out_dim, act=nn.GELU):
        """
        Mel Spectrogram을 입력으로 사용하는 Coordinator
        """
        super(Coordinator, self).__init__()
        self.encoder = EncoderManual(mel_bins, out_dim, act=act)
        self.decoder = DecoderManual(out_dim, mel_bins, frames, act=act)

    def forward(self, x):
        # Mel 데이터를 입력으로 받아 Encoder 통과
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z