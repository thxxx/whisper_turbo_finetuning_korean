#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper 파인튜닝 (간결 버전)
- get_dataset() 가 (train_ds, eval_ds) 를 반환한다고 가정
- 매 학습의 10%, 20%, ... 지점마다 체크포인트 저장
- 매 검증 시, eval 샘플 10개 (오디오+GT 텍스트+예측 텍스트) 저장 & TensorBoard 로깅
- 지나치게 긴 출력 방지: max_new_tokens, no_repeat_ngram_size, repetition_penalty, early_stopping 등 설정

사용 예)
    python train_whisper_finetune.py \
      --model_name openai/whisper-small \
      --output_dir ./whisper-small-ko \
      --epochs 3 \
      --train_bs 8 --eval_bs 8
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# 사용자 유틸 (필요 시 교체)
from notebooks.utils import wer, cer
from get_data import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo"))
    p.add_argument("--output_dir", type=str, default="./whisper-finetune")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--audio_column_name", type=str, default="audio")  # Emilia 류면 "mp3" 등으로 바꾸기
    p.add_argument("--language", type=str, default=os.getenv("LANGUAGE", "korean"))
    p.add_argument("--task", type=str, default=os.getenv("TASK", "transcribe"))  # 'translate' 가능
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_eval_samples", type=int, default=10)
    return p.parse_args()


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = " ".join(s.strip().split())
    return s


@dataclass
class OnTheFlyCollator:
    processor: WhisperProcessor
    audio_col: str = "audio"
    max_target_len: int = 256

    def __call__(self, feats: List[Dict[str, Any]]):
        # Audio → input_features
        arrays = [f[self.audio_col]["array"] for f in feats]
        srs = [f[self.audio_col].get("sampling_rate", 16000) for f in feats]
        # Whisper feature extractor는 샘플레이트 지정 필요 (혼재 시 첫 값 사용)
        inputs = self.processor.feature_extractor(
            arrays, sampling_rate=int(srs[0]), return_tensors="pt"
        )
        # Text → labels
        texts = [normalize_text(f["text"]) for f in feats]
        tok = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_target_len
        )
        lab_ids = tok.input_ids
        attnm = tok.attention_mask
        # pad_id = self.processor.tokenizer.pad_token_id
        # lab_ids = lab_ids.masked_fill(lab_ids == pad_id, -100)
        lab_ids = lab_ids.masked_fill(attnm.ne(1), -100)

        return {"input_features": inputs["input_features"], "labels": lab_ids}


class TenCheckpointAndEvalSamples(TrainerCallback):
    """10% 단위 저장 + 검증 시 샘플 오디오/텍스트 로깅"""

    def __init__(self, processor: WhisperProcessor, eval_ds, audio_col: str, out_dir: str, num_eval_samples: int = 10):
        self.processor = processor
        self.eval_ds = eval_ds
        self.audio_col = audio_col
        self.out_dir = out_dir
        self.num_eval = max(1, num_eval_samples)
        tb_dir = os.path.join(out_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.saved_marks = set()  # {1..10}

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.flush()
        self.writer.close()

    # 1) 10% 마다 저장
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.max_steps:
            return
        frac = state.global_step / state.max_steps
        for k in range(1, 11):  # 10%, 20%, ... 100%
            if frac >= (k / 10.0) and k not in self.saved_marks:
                ckpt_dir = os.path.join(self.out_dir, f"checkpoint_{k*10:03d}pct")
                os.makedirs(ckpt_dir, exist_ok=True)
                if model is not None:
                    model.save_pretrained(ckpt_dir)
                self.processor.save_pretrained(ckpt_dir)
                self.saved_marks.add(k)
        return

    # 2) 평가 때 샘플 10개 로깅/저장
    @torch.inference_mode()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or len(self.eval_ds) == 0:
            return
        model_device = next(model.parameters()).device
        model.eval()

        # 고정된 간격으로 뽑기 (재현성)
        n = min(self.num_eval, len(self.eval_ds))
        idxs = np.linspace(0, len(self.eval_ds) - 1, n, dtype=int).tolist()
        step_dir = os.path.join(self.out_dir, f"samples/step_{state.global_step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        rows = []

        for i, idx in enumerate(idxs):
            ex = self.eval_ds[idx]
            wav = ex[self.audio_col]["array"]
            sr = int(ex[self.audio_col].get("sampling_rate", 16000))

            feats = self.processor.feature_extractor(wav, sampling_rate=sr).input_features
            feats = torch.tensor(feats, dtype=torch.float32, device=model_device)

            gen_ids = model.generate(
                feats,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            pred = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

            # TensorBoard 로깅
            self.writer.add_audio(f"eval_sample/{i}/audio", wav.astype(np.float32), global_step=state.global_step, sample_rate=sr)
            self.writer.add_text(
                f"eval_sample/{i}/text",
                f"GT: {ex['text']}\nPRD: {pred}",
                global_step=state.global_step,
            )

            # 파일로도 저장 (.wav / .jsonl)
            wav_path = os.path.join(step_dir, f"sample_{i:02d}.wav")
            _save_wav(wav_path, wav, sr)
            rows.append({"index": idx, "gt": ex["text"], "pred": pred.strip()})

        with open(os.path.join(step_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# WAV 저장(표준 라이브러리만 사용)
def _save_wav(path: str, y: np.ndarray, sr: int):
    import wave
    y = np.asarray(y)
    if y.ndim > 1:
        # 다채널이면 모노 평균
        y = np.mean(y, axis=1)
    y = np.clip(y, -1.0, 1.0)
    y_i16 = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(y_i16.tobytes())

# ---- 메인 ----
def main():
    args = parse_args()

    # 데이터셋 로드 (사용자 정의)
    train_ds, eval_ds = get_dataset()

    # 프로세서 & 모델
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # 학습 중 캐시 비활성화 (GC/DP 이슈 회피)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 언어/태스크 힌트 + 긴 출력 방지용 기본 generate 설정
    try:
        model.generation_config.language = args.language
        model.generation_config.task = args.task
    except Exception:
        pass
    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    except Exception:
        pass

    gen_conf = model.generation_config
    gen_conf.max_new_tokens = args.max_new_tokens
    gen_conf.no_repeat_ngram_size = 3
    gen_conf.repetition_penalty = 1.05
    gen_conf.early_stopping = False
    # gen_conf.num_beams = 3
    gen_conf.eos_token_id = processor.tokenizer.eos_token_id
    gen_conf.pad_token_id = processor.tokenizer.eos_token_id

    collator = OnTheFlyCollator(processor, audio_col=args.audio_column_name)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        label_ids = np.where(label_ids != -100, label_ids, pad_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        sw = 0
        sc = 0
        for i in range(len(pred_str)):
            sw += wer(pred_str[i], label_str[i])
            sc += cer(pred_str[i], label_str[i])
        
        return {"wer": sw/len(pred_str), "cer": sc/len(pred_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        report_to=["tensorboard"],
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=50,
        save_total_limit=10,
        remove_unused_columns=False,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,
        predict_with_generate=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # DDP는 Trainer 가 자동 설정(LOCAL_RANK 등) 사용
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            TenCheckpointAndEvalSamples(
                processor=processor,
                eval_ds=eval_ds,
                audio_col=args.audio_column_name,
                out_dir=args.output_dir,
                num_eval_samples=args.num_eval_samples,
            )
        ],
    )

    print("\n[Train] starting...\n")
    trainer.train()

    print("\n[Eval] final evaluation...\n")
    metrics = trainer.evaluate()
    print("Final metrics:", metrics)

    print("\n[Save] final model...\n")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # 간단 추론 데모
    if len(eval_ds) > 0:
        ex = eval_ds[0]
        wav = ex[args.audio_column_name]["array"]
        sr = int(ex[args.audio_column_name].get("sampling_rate", 16000))
        feats = processor.feature_extractor(wav, sampling_rate=sr).input_features
        feats = torch.tensor(feats, dtype=torch.float32, device=next(model.parameters()).device)
        with torch.no_grad():
            out_ids = model.generate(torch.as_tensor(feats), do_sample=False)
        hyp = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        print("GT :", ex["text"]) 
        print("PRD:", hyp)


if __name__ == "__main__":
    main()
