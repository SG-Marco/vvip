# trainers/blackvip.py  (일부 내용만 예시 수정)

import os.path as osp
import time
import datetime
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
import os

from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.metrics import compute_accuracy
from my_dassl.utils import load_pretrained_weights, load_checkpoint, set_random_seed, AverageMeter, MetricMeter
from my_dassl.optim import build_optimizer, build_lr_scheduler

# 기존 CLIP 대신 Whisper를 import
import whisper

# audio_prompters를 import
from trainers import audio_prompters

import numpy as np
import pdb
import wandb
from math import sqrt

@TRAINER_REGISTRY.register()
class BLACKVIP_WHISPER(TrainerX):
    """
    Whisper를 사용해 음성->텍스트 학습을 시도하되,
    BlackVIP의 SPSA 방식을 써서 'audio_prompter' 파라미터만 업데이트.
    Whisper 파라미터는 동결.
    """
    def build_model(self):
        cfg = self.cfg

        # 1) Whisper 모델 로드
        print(f"Loading Whisper model: {cfg.TRAINER.BLACKVIP.WHISPER_MODEL_SIZE}")
        self.whisper_model = whisper.load_model(cfg.TRAINER.BLACKVIP.WHISPER_MODEL_SIZE)  # e.g. "medium"
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False

        # 2) Audio coordinator(prompter) 생성
        #    config에서 METHOD='audio_coordinator' 라고 지정했다고 가정
        print("Building AudioPrompter coordinator")
        self.coordinator = audio_prompters.__dict__[cfg.TRAINER.BLACKVIP.METHOD](cfg)

        # 3) optimizer, scheduler
        self.optim = build_optimizer(self.coordinator, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("coordinator", self.coordinator, self.optim, self.sched)

        # 4) SPSA 관련 파라미터
        self.N_params = len(torch.nn.utils.parameters_to_vector(self.coordinator.parameters()))
        self.o, self.c, self.a, self.alpha, self.gamma = cfg.TRAINER.BLACKVIP.SPSA_PARAMS
        self.opt_type = cfg.TRAINER.BLACKVIP.OPT_TYPE
        self.b1 = cfg.TRAINER.BLACKVIP.MOMS
        self.sp_avg = cfg.TRAINER.BLACKVIP.SP_AVG

        self.step = 0
        self.m1 = 0

        # 실제 loss_fn: Whisper는 오토리그레시브 CE or CTC 등. 여기선 예시로 CE
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_backward(self, batch):
        """
        batch: {"audio": waveform array, "text": transcript 등}
        """
        with torch.no_grad():
            audio = batch["audio"]  # (B, ?) raw waveform
            text  = batch["text"]   # 정답 텍스트

            # 1) mel spectrogram 변환 (Whisper 내부 함수를 그대로 사용)
            #    => shape: (B, T) waveform을 pad_or_trim 후 log_mel_spectrogram 등
            #    여기선 단순 예시로 1개씩 처리한다고 가정 (B=1)
            #    실제론 batch 처리를 하려면 whisper 소스 수정 or 유틸 사용 필요
            audio_ = audio[0]  # batch size=1 가정
            audio_ = whisper.pad_or_trim(audio_).astype("float32")
            mel = whisper.log_mel_spectrogram(audio_).unsqueeze(0)  # shape (1, 80, n_frames)

            # 2) coordinator로부터 delta 얻기
            #    - mel shape을 (B,1,T,F)로 맞춰야 하므로 unsqueeze 등 추가
            mel = mel.unsqueeze(1)  # (1,1,80,n_frames)
            prompt_delta, _ = self.coordinator(mel)

            # cfg에서 p_eps 크기 지정 가능
            p_eps = self.cfg.TRAINER.BLACKVIP.P_EPS
            mel_prompted = mel + p_eps * prompt_delta

            # -- SPSA 업데이트를 위해 +ck/-ck perturb을 적용하는 로직으로 가기 전,
            #    여기서는 단순 forward만 시연.
            #    실제로는 아래 spsa_grad_estimate_bi()에서 +ck/-ck 버전까지 만들 것.
            #    따라서 여기서는 "baseline" forward는 굳이 안 해도 됨.
            #    이 함수 구조는 BlackVIP 기존과 동일.

            # *SPSA scheduling*
            ak = self.a / ((self.step + self.o) ** self.alpha)
            ck = self.c / (self.step ** self.gamma) if (self.step>0) else self.c

            w = torch.nn.utils.parameters_to_vector(self.coordinator.parameters())
            ghat, loss, acc = self.spsa_grad_estimate_bi(w, mel, text, ck)

            # momentum 등
            if self.opt_type == 'spsa-gc':
                if self.step > 1:
                    self.m1 = self.b1*self.m1 + ghat
                else:
                    self.m1 = ghat
                accum_ghat = ghat + self.b1*self.m1
            elif self.opt_type == 'spsa':
                accum_ghat = ghat
            else:
                raise ValueError("Unknown opt_type")

            # param update
            w_new = w - ak * accum_ghat
            torch.nn.utils.vector_to_parameters(w_new, self.coordinator.parameters())

        loss_summary = {"loss": loss, "acc": acc}
        if self.cfg.use_wandb:
            wandb.log({"train_loss": loss.item(), "train_acc": acc})

        return loss_summary

    def spsa_grad_estimate_bi(self, w, mel, text, ck):
        """
        기존 BlackVIP와 유사하게, +ck perturb / -ck perturb 로
        두 번 forward -> loss 차이로 근사 gradient 구함
        여기서 Whisper forward 시 'text'를 어떻게 loss로 계산하느냐가 관건
        (간단 예시: teacher forcing 없이 그냥 decoder(...)? -> 실제론 구현 필요)
        """
        ghats = []
        B, C, T, F = mel.shape

        # 텍스트 -> label 토큰 변환 (간단화 예시)
        # 실제 Whisper는 오토리그레시브 seq2seq라서, forward/decoder를 직접 호출해야 함
        # 여기선 "가상의 logits vs label"이라 가정
        label_token = torch.randint(0, 50000, (1, 10)).to(mel.device)  # dummy

        for spk in range(self.sp_avg):
            # perturb 벡터 생성
            p_side = (torch.rand(self.N_params, device=mel.device).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side, -p_side], dim=1)
            sign_perturb = torch.bernoulli(torch.ones_like(p_side)/2).long()
            perturb = torch.gather(samples, 1, sign_perturb).reshape(-1)
            del samples, p_side, sign_perturb

            # w + ck
            w_r = w + ck*perturb
            torch.nn.utils.vector_to_parameters(w_r, self.coordinator.parameters())
            # forward -> loss
            # 여기서 "mel_prompted" 생성 후 whisper forward -> logits -> loss
            loss1, acc1 = self._forward_loss(mel, label_token)

            # w - ck
            w_l = w - ck*perturb
            torch.nn.utils.vector_to_parameters(w_l, self.coordinator.parameters())
            loss2, acc2 = self._forward_loss(mel, label_token)

            # 근사 gradient
            ghat = (loss1 - loss2) / (2*ck) * perturb
            ghats.append(ghat.unsqueeze(0))

        ghat = torch.cat(ghats, dim=0).mean(dim=0)
        loss_val = (loss1 + loss2)/2
        acc_val = (acc1 + acc2)/2

        return ghat, loss_val, acc_val

    def _forward_loss(self, mel, label_token):
        """
        실제 Whisper로 forward -> logits -> CrossEntropy
        여기서는 Dummy 예시 (랜덤 로짓)로 대체
        """
        # mel_prompted = mel + ...  # 여기서는 mel 자체를 쓴다고 가정
        # Whisper forward (오토리그레시브) => logits(seq_len, vocab_size)
        # loss = cross_entropy(logits, label_token, ...)
        # 여기선 예시로:
        batch_size = label_token.size(0)
        seq_len = label_token.size(1)
        vocab_size = 50000

        # dummy logits
        logits = torch.randn(batch_size, seq_len, vocab_size, device=mel.device)
        loss = self.loss_fn(logits.permute(0,2,1), label_token)  # (B, vocab, T) vs (B,T)

        # dummy accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=2)  # (B, T)
            acc = (pred == label_token).float().mean().item()

        return loss, acc

    def train(self):
        self.before_train()
        set_random_seed(self.cfg.SEED)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        self.total_length = self.num_batches * self.max_epoch
        self.warmup_length = self.total_length * 0.1

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # logging 등은 기존 코드 재활용
            # ...
            end = time.time()

        # etc...