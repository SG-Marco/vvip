import torch
import torch.nn as nn
from transformers import ViTModel, WhisperForConditionalGeneration

class Coordinator(nn.Module):
    def __init__(self, encoder_name="google/vit-base-patch16-224-in21k", mel_bins=128, frames=3000, hidden_dim=768):
        super(Coordinator, self).__init__()

        # 🔹 **Pretrained Encoder (Freeze)**
        self.encoder = ViTModel.from_pretrained(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False  # **Encoder는 Freeze**

        # 🔹 **Trigger Vector 추가 (학습 가능)**
        self.p_trigger = nn.Parameter(torch.randn(1, hidden_dim))  # **Trigger Vector**

        # 🔹 **Transformer 기반 Decoder**
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)  # 6층 Transformer Decoder

        # 🔹 **Decoder 출력 조정 (Mel Spectrogram 형태)**
        self.output_layer = nn.Linear(hidden_dim, mel_bins * frames)  # 최종 Mel Spectrogram 복원

    def forward(self, x):
        """
        x: (batch_size, channels, height, width) → Transformer Encoder 입력
        """
        # **1. Pretrained Encoder 통해 Feature 추출**
        encoder_outputs = self.encoder(x).last_hidden_state  # (batch_size, 197, hidden_dim)
        
        # **2. Trigger Vector 추가 (Reprogramming 효과)**
        z_with_trigger = encoder_outputs[:, 0, :] + self.p_trigger  # (batch_size, hidden_dim)

        # **3. Transformer Decoder 통해 Mel Spectrogram 복원**
        z_with_trigger = z_with_trigger.unsqueeze(0)  # (1, batch_size, hidden_dim) → Transformer Decoder 입력
        mel_features = self.decoder(z_with_trigger, encoder_outputs.transpose(0, 1))  # (batch_size, hidden_dim)
        
        # **4. 최종 Mel Spectrogram 형태로 변환**
        mel_output = self.output_layer(mel_features)  # (batch_size, mel_bins * frames)
        mel_output = mel_output.view(-1, 1, 128, 3000)  # (batch_size, 1, mel_bins, frames)

        return z_with_trigger, mel_output