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
from typing import Any, Dict, List, Union

import numpy as np
import torch
import evaluate
import librosa

from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
)

# -----------------------------
# 1) 하이퍼파라미터 & 인자 파서
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()

    # 데이터셋
    parser.add_argument("--audio_column_name", type=str, default="mp3")
    parser.add_argument("--text_column_name", type=str, default="text")

    # 모델 & 언어
    parser.add_argument("--model_name", type=str, default=os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo"))
    parser.add_argument("--language", type=str, default=os.getenv("LANGUAGE", "korean"))     # 'korean'
    parser.add_argument("--task", type=str, default=os.getenv("TASK", "transcribe"))         # 'transcribe' or 'translate'
    parser.add_argument("--sampling_rate", type=int, default=16000)

    # 학습 설정
    parser.add_argument("--output_dir", type=str, default="./whisper-finetune")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", help="fp16 사용 (CUDA)")
    parser.add_argument("--bf16", action="store_true", help="bf16 사용 (Ampere+)")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1, help=">0이면 epoch 대신 steps 기준 종료")

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


# ------------------------------------
# 2) 텍스트 전처리(간단 버전: 한국어 위주)
# ------------------------------------
def normalize_text(txt: str) -> str:
    """
    - 소문자화, 양끝 공백 제거
    - 불필요한 중복 공백 제거
    - (옵션) 기호 단순화 등
    NOTE: 프로젝트 특성에 맞춰 적절히 확장하세요.
    """
    if txt is None:
        return ""
    t = txt.strip()
    # Whisper는 케이스/구두점 학습도 가능하므로, 너무 aggressive 하게 제거하지 않습니다.
    t = " ".join(t.split())
    return t


# ------------------------------------
# 3) DataCollator: 패딩/라벨 마스크링
# ------------------------------------
@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        inp = [{"input_features": f["input_features"]} for f in features]
        lab = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(lab, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


# -----------------------------
# 4) 메인
# -----------------------------
def main():
    args = get_args()

    # 4-1) 데이터셋 로드
    train_ds = load_dataset("amphion/Emilia-Dataset", data_files={"train": "Emilia/KO/*.tar"}, split="train")
    eval_ds = load_dataset("amphion/Emilia-Dataset", data_files={"validation": "Emilia/KO/*.tar"}, split="validation")

    # 오디오 열을 Whisper 샘플레이트(16k)로 캐스팅
    # train_ds = train_ds.cast_column(args.audio_column_name, Audio(sampling_rate=args.sampling_rate))
    # eval_ds = eval_ds.cast_column(args.audio_column_name, Audio(sampling_rate=args.sampling_rate))

    # 4-2) 프로세서 & 모델
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

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

    # 4-3) 전처리 함수
    def prepare_example(batch):
        # 1) 오디오 -> log-mel
        audio = batch[args.audio_column_name]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=args.sampling_rate
        ).input_features[0]

        # 2) 텍스트 정리 -> 토크나이즈(라벨)
        text = normalize_text(batch['json']['text'])
        labels = processor.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    # 사전 변환(메모리 고려 시 batched=True + num_proc 조절 가능)
    train_proc = train_ds.map(prepare_example, remove_columns=train_ds.column_names)
    eval_proc = eval_ds.map(prepare_example, remove_columns=eval_ds.column_names)

    # 4-4) 데이터 콜레이터
    data_collator = DataCollatorSpeechSeq2Seq(processor=processor)

    # 4-5) 지표: WER / CER
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        # pred: PredictionOutput with predictions (logits->ids) and label_ids(-100 포함)
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):  # 일부 버전에서 (logits,) 구조
            pred_ids = pred_ids[0]

        # label_ids에서 -100을 pad_token_id로 복구 후 디코딩
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # 4-6) 학습 인자
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["none"],  # 필요 시 'tensorboard', 'wandb'
        max_steps=args.max_steps if args.max_steps > 0 else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # 4-7) Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_proc,
        eval_dataset=eval_proc,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,  # 로그/저장 시 필요
        compute_metrics=compute_metrics if args.eval_strategy != "no" else None,
    )

    print("\n\nStart traing\n\n")

    # 4-8) 학습 & 평가 & 저장
    trainer.train()
    metrics = trainer.evaluate() if args.eval_strategy != "no" else {}
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
