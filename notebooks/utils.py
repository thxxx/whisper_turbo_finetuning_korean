import unicodedata
import re
from typing import List, Dict
import random

# -------------------------
# 1) 텍스트 정규화
# -------------------------
_ALNUM_KO = re.compile(r"[a-z0-9\uAC00-\uD7A3 ]")  # 영/숫/한글음절/공백만 허용

def normalize_ko_text(s: str) -> str:
    """
    - 유니코드 정규화(NFKC 후 NFC)
    - 영문 소문자화
    - 문장부호/기호 제거(콤마 등)
    - 다중 공백 축소
    """
    if s is None:
        return ""

    # 1) 호환문자 정리 + 조합형 음절 정규화
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFC", s)

    # 2) 소문자화(영문)
    s = s.lower()

    # 3) 허용 문자만 남기기: 한글 음절, a-z, 0-9, space
    #    -> 문장부호(, . ! ? 등) 및 기타 기호 제거
    kept = []
    for ch in s:
        if _ALNUM_KO.fullmatch(ch):
            kept.append(ch)
        else:
            # 문장부호/기호/제어문자 등은 제거
            pass
    s = "".join(kept)

    # 4) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------
# 2) 편리한 토크나이저
# -------------------------
def tokens_word(s: str) -> List[str]:
    s = normalize_ko_text(s)
    if not s:
        return []
    return s.split()

def tokens_char(s: str, keep_space: bool = False) -> List[str]:
    """
    CER/문자기반 토큰화: 기본은 공백 제거.
    keep_space=True면 공백도 문자로 취급.
    """
    s = normalize_ko_text(s)
    if not s:
        return []
    return list(s if keep_space else s.replace(" ", ""))


# -------------------------
# 3) Levenshtein 거리 (편집거리)
# -------------------------
def levenshtein(a: List[str], b: List[str]) -> int:
    """
    표준 DP 구현 (메모리 O(min(n,m)) 최적화 없이도 작은 텍스트에는 충분)
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 삭제
                dp[i][j-1] + 1,      # 삽입
                dp[i-1][j-1] + cost  # 교체(일치면 0)
            )
    return dp[n][m]


# -------------------------
# 4) WER / CER
# -------------------------
def wer(ref: str, hyp: str, unit: str = "word") -> float:
    """
    unit='word' : 공백 단어 단위 WER (기본)
    unit='char' : 문자(음절) 단위 WER (띄어쓰기 편차 완화용)
    """
    if unit not in ("word", "char"):
        raise ValueError("unit must be 'word' or 'char'")

    ref_tokens = tokens_word(ref) if unit == "word" else tokens_char(ref, keep_space=False)
    hyp_tokens = tokens_word(hyp) if unit == "word" else tokens_char(hyp, keep_space=False)

    if len(ref_tokens) == 0:
        # ref가 비었으면: hyp도 비었으면 0, 아니면 1
        return 0.0 if len(hyp_tokens) == 0 else 1.0

    dist = levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def cer(ref: str, hyp: str) -> float:
    """
    공백 제거 문자 단위 CER
    """
    ref_tokens = tokens_char(ref, keep_space=False)
    hyp_tokens = tokens_char(hyp, keep_space=False)

    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0

    dist = levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)

def clean_transcript(example):
    text = example["text"]

    # 1) n/, o/ 같은 접두어 제거 (문장 어디서든 등장하면 삭제)
    text = re.sub(r"\b[no]/\s*", "", text)

    # 2) (foo)/(bar) → foo 또는 bar 중 랜덤 선택
    def repl(match):
        opts = match.group(0)[1:-1].split(")/(")  # 괄호 제거 후 분리
        # return opts[0]
        return random.choice(opts)
    text = re.sub(r"\([^()]+?\)/\([^()]+?\)", repl, text)

    # 3) 발음 변이 표기 (예: (바)) → 통째로 제거
    text = re.sub(r"\([^()/]+\)", "", text)

    example["text"] = text.strip()
    return example