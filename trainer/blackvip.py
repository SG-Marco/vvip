# VVIP/trainers/blackvip.py

import os
import time
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

# whisper import
import whisper
from torch.utils.data import DataLoader, Dataset

from audio_prompters import audio_coordinator  # (or your custom AudioPrompter)
# from .audio_prompting_whisper import AudioPromptingWhisper  # alternative approach

class SimpleWaveformDataset(Dataset):
    """
    예시용: (audio, text) 형태를 반환
    """
    def __init__(self, list_of_samples):
        """
        list_of_samples: [ (waveform: np.array, text: str), ... ]
        """
        self.samples = list_of_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_arr, text_str = self.samples[idx]
        return {"audio": audio_arr, "text": text_str}

class BLACKVIP_WHISPER:
    def __init__(self, cfg):
        self.cfg = cfg
        # 1) Whisper 모델 (블랙박스)
        self.whisper_model = whisper.load_model(cfg["WHISPER_SIZE"])  # "medium" etc
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False

        # 2) AudioPrompter
        print("Building AudioPrompter coordinator")
        self.coordinator = audio_coordinator(cfg)  # => returns AudioPrompter(cfg)
        self.coordinator.train()

        # 3) SPSA params
        self.N_params = sum(p.numel() for p in self.coordinator.parameters())
        self.o, self.c, self.a, self.alpha, self.gamma = cfg["SPSA_PARAMS"]
        self.opt_type = cfg["OPT_TYPE"]  # "spsa" or "spsa-gc"
        self.b1 = cfg["MOMS"]
        self.sp_avg = cfg["SP_AVG"]

        self.step = 0
        self.m1 = 0
        self.loss_fn = nn.CrossEntropyLoss()

        # 4) dataset / dataloader
        data_list = cfg["DATA"]  # list of (audio_arr, text_str)
        self.dataset = SimpleWaveformDataset(data_list)
        self.train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.max_epoch = cfg["EPOCHS"]

    def train(self):
        print("[BLACKVIP_WHISPER] Start Training with SPSA!")
        start_time = time.time()
        for epoch in range(self.max_epoch):
            for batch in self.train_loader:
                self.step += 1
                # forward & spsa update
                loss_summary = self.forward_backward(batch)
                if self.step % 10 == 0:
                    print(f"step={self.step} loss={loss_summary['loss']:.4f} acc={loss_summary['acc']:.4f}")

        elapsed = time.time() - start_time
        print(f"Training done. Elapsed {elapsed/60:.1f} min")

    def forward_backward(self, batch):
        with torch.no_grad():
            audio = batch["audio"][0].numpy()  # shape (?,)
            text = batch["text"][0]

            audio_ = whisper.pad_or_trim(audio).astype("float32")
            mel = whisper.log_mel_spectrogram(audio_)  # shape (80, frames)
            mel = mel.unsqueeze(0).unsqueeze(0)  # => (B=1, C=1, 80, frames)

            # SPSA schedule
            ak = self.a / ((self.step + self.o)**self.alpha)
            ck = self.c / (self.step**self.gamma + 1e-8)

            w = torch.nn.utils.parameters_to_vector(self.coordinator.parameters()).cpu()
            ghat, loss, acc = self.spsa_grad_estimate_bi(w, mel, text, ck)

            # momentum
            if self.opt_type == 'spsa-gc':
                if self.step > 1:
                    self.m1 = self.b1*self.m1 + ghat
                else:
                    self.m1 = ghat
                accum_ghat = ghat + self.b1*self.m1
            elif self.opt_type == 'spsa':
                accum_ghat = ghat
            else:
                raise ValueError("Unknown opt_type in forward_backward")

            w_new = w - ak * accum_ghat
            torch.nn.utils.vector_to_parameters(w_new, self.coordinator.parameters())

        return {"loss": loss, "acc": acc}

    def spsa_grad_estimate_bi(self, w, mel, text, ck):
        ghats = []
        B, C, T, F = mel.shape

        # text -> label_token (dummy example)
        label_token = torch.randint(0, 50000, (1, 10))

        for spk in range(self.sp_avg):
            # +- perturb
            perturb = torch.bernoulli(torch.empty_like(w).uniform_(0,1))
            perturb = torch.where(perturb>0.5, torch.tensor(1.0), torch.tensor(-1.0))

            w_r = w + ck*perturb
            torch.nn.utils.vector_to_parameters(w_r, self.coordinator.parameters())
            loss1, acc1 = self._forward_loss(mel)

            w_l = w - ck*perturb
            torch.nn.utils.vector_to_parameters(w_l, self.coordinator.parameters())
            loss2, acc2 = self._forward_loss(mel)

            ghat_sp = (loss1 - loss2) / (2*ck) * perturb
            ghats.append(ghat_sp.unsqueeze(0))

        ghat = torch.cat(ghats, dim=0).mean(dim=0)
        loss_val = (loss1 + loss2).item()/2
        acc_val = (acc1 + acc2)/2

        return ghat, loss_val, acc_val

    def _forward_loss(self, mel):
        """
        실제론 Whisper forward -> 로짓 -> CrossEntropy
        여기선 dummy
        """
        batch_size = 1
        seq_len = 10
        vocab_size = 50000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        loss = self.loss_fn(logits.permute(0,2,1), torch.randint(0, vocab_size,(batch_size,seq_len)))
        with torch.no_grad():
            pred = logits.argmax(dim=2)
            acc = (pred==0).float().mean().item()  # dummy
        return loss, acc

    def save_delta(self, path="spsa_audio_delta.pth"):
        torch.save(self.coordinator.state_dict(), path)
        print(f"[BLACKVIP_WHISPER] Saved coordinator (delta) to {path}")

    def load_delta(self, path="spsa_audio_delta.pth"):
        self.coordinator.load_state_dict(torch.load(path))
        print(f"[BLACKVIP_WHISPER] Loaded coordinator (delta) from {path}")