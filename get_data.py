from datasets import load_dataset, Audio
from datasets import concatenate_datasets
import numpy as np
from notebooks.utils import clean_transcript

def get_dataset():
    # ds4 = load_dataset("Junhoee/STT_Korean_Dataset")
    # splits4 = ds4['train'].train_test_split(test_size=0.1, seed=42)  # seed 고정하면 재현성 O
    # train_ds4 = splits4["train"].shuffle(seed=42).select(range(50000))
    # val_ds4   = splits4["test"].shuffle(seed=42).select(range(1500))
    # train_ds4 = train_ds4.rename_column("transcripts", "text")
    # val_ds4 = val_ds4.rename_column("transcripts", "text")

    ds1 = load_dataset("Junhoee/STT_Korean_Dataset")
    splits1 = ds1['train'].train_test_split(test_size=0.1, seed=42)  # seed 고정하면 재현성 O
    train_ds1 = splits1["train"].shuffle(seed=42) # 총 16만개
    val_ds1   = splits1["test"].shuffle(seed=42).select(range(1500))
    train_ds1 = train_ds1.rename_column("transcripts", "text")
    val_ds1 = val_ds1.rename_column("transcripts", "text")

    ds2 = load_dataset("kresnik/zeroth_korean")
    splits2 = ds2["train"].train_test_split(test_size=0.1, seed=42)  # seed 고정하면 재현성 O
    train_ds2 = splits2["train"]
    val_ds2   = splits2["test"].shuffle(seed=42).select(range(1500))
    # audio, text

    ds3 = load_dataset("idiotDeveloper/koreanTelephone")
    ds3 = ds3.cast_column("audio", Audio(decode=True, sampling_rate=16000))
    arr = np.array(ds3['train']["transcripts"])
    mask = np.array([not ("((" in t or "))" in t) for t in arr])
    ds3['train'] = ds3['train'].select(np.where(mask)[0])

    arr = np.array(ds3['test']["transcripts"])
    mask = np.array([not ("((" in t or "))" in t) for t in arr])
    ds3['test'] = ds3['test'].select(np.where(mask)[0])

    train_ds3 = ds3['train'].shuffle(seed=42).select(range(160000))
    val_ds3 = ds3['test'].shuffle(seed=42).select(range(1500))

    train_ds3 = train_ds3.rename_column("transcripts", "text")
    val_ds3 = val_ds3.rename_column("transcripts", "text")
    # audio, transcripts
    train_ds3 = train_ds3.map(clean_transcript)
    val_ds3 = val_ds3.map(clean_transcript)

    train_ds = concatenate_datasets([train_ds1, train_ds2, train_ds2, train_ds3]) # 총 17만개?
    # val_ds = concatenate_datasets([val_ds1]) # 총 4500개
    val_ds = concatenate_datasets([val_ds1, val_ds2, val_ds3]) # 총 4500개

    return train_ds, val_ds




