import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import torchaudio
from tqdm import tqdm
from notebooks.utils import clean_transcript
# from notebooks.utils import wer, cer, clean_transcript
import argparse
import numpy as np

import Levenshtein as Lev

# Character Error Rate
def cer(ref: str, hyp: str) -> float:
    """
    CER = (삽입 + 삭제 + 교체) / 정답 글자 수
    """
    ref = ref.replace(" ", "")  # CER은 보통 공백 제외
    hyp = hyp.replace(" ", "")
    distance = Lev.distance(ref, hyp)
    return distance / max(1, len(ref))

# Word Error Rate
def wer(ref: str, hyp: str) -> float:
    """
    WER = (삽입 + 삭제 + 교체) / 정답 단어 수
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    distance = Lev.distance(" ".join(ref_words), " ".join(hyp_words))
    return distance / max(1, len(ref_words))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_zeroth", action="store_true")
    parser.add_argument("--no_emilia", action="store_true")
    parser.add_argument("--no_telephone", action="store_true")
    return parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2"
model_id = "/home/khj6051/whisper/whisper-turbo-ko-third/checkpoint_100pct"
# model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

if __name__ == "__main__":
    args = get_args()

    out_file = "wer_whisper_mine.txt"
    def write_result(wers, cers, name):
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"[{name}] - WER: {sum(wers)/len(wers)}, CER: {sum(cers)/len(cers)}\n")

    if not args.no_zeroth:
        ds = load_dataset("kresnik/zeroth_korean")["train"]
        splits = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = splits["train"]
        val_ds   = splits["test"]

        wers = []
        cers = []

        pbar = tqdm(len(val_ds))
        print("\n\nStart Zeroth\n\n")
        for i, data in enumerate(val_ds):
            pbar.update(1)
            array = torch.tensor(data['audio']['array']).to(device)
            sr = data['audio']['sampling_rate']
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
                array = resampler(array)
            result = pipe(array)
            rt = result['text'].strip()

            nw = wer(rt, data['text'])
            nc = cer(rt, data['text'])
            wers.append(nw)
            cers.append(nc)
            pbar.set_description(f"[{i+1}th] - WER: {sum(wers)/len(wers):.4f}, CER: {sum(cers)/len(cers):.4f}")

            # print("\n", rt)
            # print(data['text'])
            # print("-"*100)
            # if nw>0:
            #     print("wrong")

            if i%500 == 499:
                write_result(wers, cers, f"Zeroth {i+1}th")
        write_result(wers, cers, "Zeroth")

    if not args.no_emilia:
        dataset = load_dataset("Junhoee/STT_Korean_Dataset")
        splits = dataset['train'].train_test_split(test_size=0.1, seed=42)  # seed 고정하면 재현성 O
        val_ds = splits["test"].select(range(2000))

        wers, cers = [], []

        pbar = tqdm(len(val_ds))
        print("\n\nStart Emilia\n\n")
        for i, data in enumerate(val_ds):
            pbar.update(1)
            array = torch.tensor(data['audio']['array']).to(device).to(pipe.dtype)
            sr = data['audio']['sampling_rate']
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(device).to(pipe.dtype)
                array = resampler(array)
            result = pipe(array)

            rt = result['text'].strip()

            nw = wer(rt, data['transcripts'])
            nc = cer(rt, data['transcripts'])
            wers.append(nw)
            cers.append(nc)

            pbar.set_description(f"[{i+1}th] - WER: {sum(wers)/len(wers):.4f}, CER: {sum(cers)/len(cers):.4f}")

            print("\n", rt)
            print(data['transcripts'])
            # print("-"*100)

            if i%500 == 499:
                write_result(wers, cers, f"Emilia {i+1}th")
        write_result(wers, cers, "Emilia")

    if not args.no_telephone:
        ds = load_dataset("idiotDeveloper/koreanTelephone")
        ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16000))

        arr = np.array(ds['test']["transcripts"])
        mask = np.array([not ("((" in t or "))" in t) for t in arr])
        ds['test'] = ds['test'].select(np.where(mask)[0])

        val_ds   = ds["test"].select(range(2000))
        val_ds = val_ds.map(clean_transcript)
        
        wers, cers = [], []

        pbar = tqdm(len(val_ds))

        print("\n\nStart Telephone\n\n")
        for i, data in enumerate(val_ds):
            pbar.update(1)
            array = torch.tensor(data['audio']['array']).to(device)
            sr = data['audio']['sampling_rate']
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
                array = resampler(array)
            result = pipe(array)
            wers.append(wer(result["text"], data['transcripts']))
            cers.append(cer(result["text"], data['transcripts']))

            pbar.set_description(f"[{i+1}th] - WER: {sum(wers)/len(wers):.4f}, CER: {sum(cers)/len(cers):.4f}")

            print("\n", result["text"])
            print(data['transcripts'])
            print("-"*100)

            if i%500 == 499:
                write_result(wers, cers, f"Telephone {i+1}th")
        write_result(wers, cers, "Telephone")
