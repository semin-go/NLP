import csv
import re
from collections import Counter
from typing import List, Dict, Optional

class SlangDetector:
    """
    세대 맞춤 신조어·은어 검출기 베이스라인 클래스
    - 사전 기반 탐지
    - 패턴 기반 탐지(자음-only, 반복 문자, ㅋㅋ/ㅠㅠ 등)
    """

    # 자음/모음만으로 이루어진 토큰 (ㄹㅇ, ㅇㅈ, ㅈㄴ 등)
    _JAMO_ONLY_RE = re.compile(r"^[ㄱ-ㅎㅏ-ㅣ]+$")

    # 같은 글자가 3번 이상 반복되는 패턴 (ㅋㅋㅋㅋ, ㅎㅎㅎ, 와아아아 등)
    _REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

    # 흔한 감탄/웃음/울음 패턴 (기본적으로 slang 취급)
    _EMO_PATTERN_RE = re.compile(r"[ㅋㅎㅠㅜ]+")  # 예: ㅋㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠ

    def __init__(
        self,
        slang_lexicon: Optional[set] = None,
        min_candidate_freq: int = 3,
        min_token_length: int = 2,
    ):
        """
        :param slang_lexicon: 사전으로 지정할 신조어 집합 (옵션)
        :param min_candidate_freq: CSV에서 자동 추출 시 최소 빈도
        :param min_token_length: 너무 짧은 토큰(1글자 등)은 필터링
        """
        self.slang_lexicon = slang_lexicon if slang_lexicon is not None else set()
        self.min_candidate_freq = min_candidate_freq
        self.min_token_length = min_token_length

    # -------------------------------
    # 1) CSV 기반 신조어 후보 자동 추출
    # -------------------------------
    def build_lexicon_from_csv(
        self,
        csv_path: str,
        source_col: str = "source",
        target_col: str = "target",
        encoding: str = "utf-8-sig",
    ):
        """
        병렬 데이터셋(CSV)에서 source에는 있지만 target에는 거의 안 나오는 토큰을
        '신조어 후보'로 간주하여 slang_lexicon에 추가한다.

        CSV 예시는 다음과 같은 구조를 따른다고 가정한다. :contentReference[oaicite:1]{index=1}
        id,generation,emotion,source,target,category,label_type
        """
        candidate_counter = Counter()

        with open(csv_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row[source_col].strip()
                tgt = row[target_col].strip()

                src_tokens = self._simple_tokenize(src)
                tgt_tokens = set(self._simple_tokenize(tgt))

                for token in src_tokens:
                    # target에 없고, 일정 길이 이상이면 slang 후보
                    if (
                        token not in tgt_tokens
                        and len(token) >= self.min_token_length
                    ):
                        candidate_counter[token] += 1

        # 빈도 기준으로 필터링
        for token, freq in candidate_counter.items():
            if freq >= self.min_candidate_freq:
                self.slang_lexicon.add(token)

    # -------------------------------
    # 2) 검출용 공개 메서드
    # -------------------------------
    def detect(self, sentence: str) -> List[Dict]:
        """
        주어진 문장에서 신조어/비표준 표현으로 추정되는 토큰 목록을 반환한다.
        반환 형식:
        [
            {
                "token": "알잘딱깔센",
                "start": 3,
                "end": 8,
                "reason": ["lexicon", "pattern_jamo"],
            },
            ...
        ]
        """
        results = []

        # 단순 공백 기준 토크나이징
        tokens = self._simple_tokenize(sentence)
        # 토큰별 위치(인덱스) 계산
        positions = self._token_spans(sentence, tokens)

        for token, (start, end) in zip(tokens, positions):
            reasons = []

            # 1) 사전 기반 검출
            if token in self.slang_lexicon:
                reasons.append("lexicon")

            # 2) 자음/모음-only 패턴 (ㄹㅇ, ㅇㅈ, ㅈㄴ 등)
            if self._JAMO_ONLY_RE.match(token):
                reasons.append("pattern_jamo")

            # 3) 반복문자 패턴 (ㅋㅋㅋㅋ, 와아아아 등)
            if self._REPEAT_CHAR_RE.search(token):
                reasons.append("pattern_repeat")

            # 4) 감탄/웃음/울음 패턴 (ㅋㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠ 등)
            if self._EMO_PATTERN_RE.fullmatch(token):
                reasons.append("pattern_emo")

            # 5) 기타: 영문+숫자 혼합, 특수문자 많이 포함된 토큰 등도 옵션으로 추가 가능

            if reasons:
                results.append(
                    {
                        "token": token,
                        "start": start,
                        "end": end,
                        "reason": reasons,
                    }
                )

        return results

    def has_slang(self, sentence: str) -> bool:
        """
        문장에 신조어/비표준 표현이 하나라도 포함되어 있는지 여부만 반환.
        """
        return len(self.detect(sentence)) > 0

    # -------------------------------
    # 3) 내부 유틸 함수들
    # -------------------------------


    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        # 1) 문장부호 제거
        text = re.sub(r"[^\w\sㄱ-힣]", "", text)

        # 2) 공백 기반 분리
        tokens = text.split()

        return [t.strip() for t in tokens if t.strip()]


    @staticmethod
    def _token_spans(text: str, tokens: List[str]) -> List[tuple]:
        """
        각 토큰이 원문 내에서 차지하는 (start, end) 인덱스를 계산한다.
        (아주 단순한 구현: 왼쪽에서부터 순차적으로 찾기)
        """
        spans = []
        idx = 0
        for token in tokens:
            start = text.find(token, idx)
            if start == -1:
                # 못 찾으면 대충 넘어가지만, 웬만하면 발생하지 않음
                start = idx
            end = start + len(token)
            spans.append((start, end))
            idx = end
        return spans
