#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단 · 단일 파일 Whisper 미세튜닝 스크립트 (한국어 기본)
- 데이터셋: HF Datasets (columns: "audio", "text")
- 모델: openai/whisper-small (환경변수로 변경 가능)
- 평가: WER/CER
- Trainer 한 방에 학습/평가/추론

사용 예)
    python train_whisper.py \
      --dataset_name mozilla-foundation/common_voice_17_0 \
      --dataset_config ko \
      --text_column_name sentence \
      --train_split train --eval_split validation \
      --output_dir ./whisper-small-ko-finetune

로컬 오디오/라벨 CSV 기반 커스텀 로더를 쓰는 경우는 아래 NOTE 참조.
"""

import os
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from notebooks.utils import wer, cer
from get_data import get_dataset

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ➍ (선택) 분산 로그 중복 방지: rank 0만 프린트하고 싶다면
def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0
if not is_main_process():
    import transformers, logging
    transformers.utils.logging.set_verbosity_error()
    logging.getLogger().setLevel(logging.ERROR)

# -----------------------------
# 1) 하이퍼파라미터 & 인자 파서
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()

    # 데이터셋
    parser.add_argument("--audio_column_name", type=str, default="")
    parser.add_argument("--text_column_name", type=str, default="text")

    # 모델 & 언어
    parser.add_argument("--model_name", type=str, default=os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo"))
    parser.add_argument("--language", type=str, default=os.getenv("LANGUAGE", "korean"))     # 'korean'
    parser.add_argument("--task", type=str, default=os.getenv("TASK", "transcribe"))         # 'transcribe' or 'translate'
    parser.add_argument("--sampling_rate", type=int, default=16000)

    # 학습 설정
    parser.add_argument("--output_dir", type=str, default="./whisper-finetune-total")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", help="fp16 사용 (CUDA)")
    parser.add_argument("--bf16", action="store_true", help="bf16 사용 (Ampere+)")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_steps", type=int, default=50000, help=">0이면 epoch 대신 steps 기준 종료")

    # 로깅/평가/체크포인트
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="steps")  # 'no'|'epoch'|'steps'
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--generation_max_length", type=int, default=225)  # whisper 기본 30s 기준

    # 기타
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)

    return parser.parse_args()

def normalize_text(txt: str) -> str:
    if txt is None:
        return ""
    t = txt.strip()
    t = " ".join(t.split())
    return t

@dataclass
class OnTheFlyCollator:
    processor: WhisperProcessor
    audio_col: str = "audio"
    def __call__(self, features):
        arrays = [f[self.audio_col]["array"] for f in features]
        inputs = self.processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt"
        )
        
        texts = [normalize_text(f['text']) for f in features]
        lab = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        # pad -> -100
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        lab = lab.masked_fill(lab == pad_id, -100)
        return {"input_features": inputs["input_features"], "labels": lab}

# -----------------------------
# 4) 메인
# -----------------------------
def main():
    args = get_args()

    # 4-1) 데이터셋 로드
    train_ds, eval_ds = get_dataset()

    # 4-2) 프로세서 & 모델
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # 🔧 훈련 중 캐시 끄기 (DP/GC 이슈 방지)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads() 

    # Whisper에 언어/태스크 힌트 설정 (신/구 API 호환)
    try:
        model.generation_config.language = args.language
        model.generation_config.task = args.task
    except Exception:
        pass
    try:
        forced_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        model.config.forced_decoder_ids = forced_ids
    except Exception:
        pass

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 사전 변환(메모리 고려 시 batched=True + num_proc 조절 가능)
    collator = OnTheFlyCollator(processor, audio_col=args.audio_column_name)

    def compute_metrics(pred):
        # pred: PredictionOutput with predictions (logits->ids) and label_ids(-100 포함)
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):  # 일부 버전에서 (logits,) 구조
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_get = wer(pred_str, label_str)
        cer_get = cer(pred_str, label_str)
        return {"wer": wer_get, "cer": cer_get}

    # 4-6) 학습 인자
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        report_to=["tensorboard"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=5,
        generation_max_length=args.generation_max_length,
        remove_unused_columns=False,
        max_steps=args.max_steps if args.max_steps > 0 else None,

        do_eval=True,   
        eval_strategy="steps",
        eval_steps=5000,
        predict_with_generate=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # DDP 권장 옵션들 ↓
        ddp_backend="nccl",                      # Linux+NVIDIA면 nccl
        ddp_find_unused_parameters=False,        # 미사용 파라미터 탐색 끔(성능/안정성↑)
        dataloader_num_workers=4,                # I/O 병렬 (서버 사양에 맞춰 조절)
    )

    print("trainer args: ", training_args)

    # 4-7) Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        # tokenizer=processor.feature_extractor,  # 로그/저장 시 필요
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n\nStart traing\n\n")

    # 4-8) 학습 & 평가 & 저장
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # 최종 저장
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # 간단 추론 예시(옵션)
    print("\n[Inference demo]")
    sample = eval_ds[0]
    wav = sample[args.audio_column_name]["array"]
    feats = processor.feature_extractor(wav, sampling_rate=args.sampling_rate).input_features
    feats = torch.tensor(feats, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(feats, max_length=args.generation_max_length)
    pred = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print("GT :", sample[args.text_column_name])
    print("PRD:", pred)


if __name__ == "__main__":
    main()
