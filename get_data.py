from datasets import load_dataset, Audio, Dataset
from datasets import concatenate_datasets
import numpy as np
from notebooks.utils import clean_transcript
import pyarrow.compute as pc  # ✨ Arrow로 dedup
import time

SR = 16000

def _cast_audio(ds, col="audio", sr=SR):
    # dataset마다 오디오 칼럼명이 다르면 여기서 맞춰주세요.
    return ds.cast_column(col, Audio(decode=True, sampling_rate=sr))

def _dedup_by_text(ds, col="text"):
    seen = set()
    keep_idx = []
    for i, t in enumerate(ds[col]):
        if t not in seen:
            seen.add(t)
            keep_idx.append(i)
    return ds.select(keep_idx)

def get_dataset():
    # ---- ds1 ----
    ds1 = load_dataset("Junhoee/STT_Korean_Dataset")
    splits1 = ds1["train"].train_test_split(test_size=0.1, seed=42)
    train_ds1 = splits1["train"].shuffle(seed=42)  # 총 16만개
    val_ds1   = splits1["test"].shuffle(seed=42).select(range(1500))
    # 스키마 통일: audio를 Audio feature로 강제
    train_ds1 = _cast_audio(train_ds1, "audio", SR)
    val_ds1   = _cast_audio(val_ds1, "audio", SR)
    # 칼럼 이름 통일
    train_ds1 = train_ds1.rename_column("transcripts", "text")
    val_ds1   = val_ds1.rename_column("transcripts", "text")
    # 텍스트 클린업
    train_ds1 = train_ds1.map(clean_transcript)
    val_ds1   = val_ds1.map(clean_transcript)

    # ---- ds2 ----
    ds2 = load_dataset("kresnik/zeroth_korean")
    splits2 = ds2["train"].train_test_split(test_size=0.1, seed=42)
    train_ds2 = splits2["train"]
    val_ds2   = splits2["test"].shuffle(seed=42).select(range(1500))
    # zeroth도 audio 스키마 통일
    train_ds2 = _cast_audio(train_ds2, "audio", SR)
    val_ds2   = _cast_audio(val_ds2, "audio", SR)
    train_ds2 = train_ds2.map(clean_transcript)
    val_ds2   = val_ds2.map(clean_transcript)

    # ---- ds3 ----
    ds3 = load_dataset("idiotDeveloper/koreanTelephone")
    ds3 = ds3.cast_column("audio", Audio(decode=True, sampling_rate=SR))

    # 괄호 처리 필터
    arr = np.array(ds3['train']["transcripts"])
    mask = np.array([not ("((" in t or "))" in t) for t in arr])
    ds3['train'] = ds3['train'].select(np.where(mask)[0])

    arr = np.array(ds3['test']["transcripts"])
    mask = np.array([not ("((" in t or "))" in t) for t in arr])
    ds3['test'] = ds3['test'].select(np.where(mask)[0])

    train_ds3 = ds3['train'].shuffle(seed=42).select(range(320000))
    val_ds3   = ds3['test'].shuffle(seed=42).select(range(1500))

    train_ds3 = train_ds3.rename_column("transcripts", "text")
    val_ds3   = val_ds3.rename_column("transcripts", "text")
    train_ds3 = train_ds3.map(clean_transcript)
    val_ds3   = val_ds3.map(clean_transcript)

    # ---- concat ----
    # 필요에 따라 가중치 주려면 반복 대신 sampling으로 비율 맞추는 걸 추천.
    train_ds = concatenate_datasets([train_ds1, train_ds1, train_ds2, train_ds2, train_ds3])
    val_ds   = concatenate_datasets([val_ds1, val_ds2, val_ds3])

    # ---- 중복 제거 (pandas X) ----
    st = time.time()
    print("dedup start", len(train_ds))
    train_ds = _dedup_by_text(train_ds, col="text")
    print("dedup end (hours) =", (time.time() - st) / 3600.0, len(train_ds))

    # ---- 마지막 섞기 ----
    train_ds = train_ds.shuffle(seed=42)

    return train_ds, val_ds
