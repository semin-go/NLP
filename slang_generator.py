# slang_generator.py
import csv
import random
import time
import json
from tqdm import tqdm
from openai import OpenAI
import os
import re

client = OpenAI(api_key="sOPENAI_API_KEY")

# -----------------------------------------------------------------------------
# 1. 슬랭 키워드 리스트 (200개+)
#    - 반드시 이 중 최소 1개 이상이 source 문장에 등장해야 샘플로 채택
#    - 너무 혐오/비하 강한 것들은 일부러 제외함 (정 필요하면 따로 수동 추가 추천)
# -----------------------------------------------------------------------------
SLANG_KEYWORDS = [
    # 네가 준 것들 + 일반적인 신조어들
    "억까", "갓생", "킹받네", "킹받다", "내돈내산", "나심비", "스몸비", "점메추", "TMI",
    "만반잘부", "알잘딱깔센", "꿀잼", "노잼", "인싸", "아싸", "월급루팡", "멍청비용",
    "갑질", "머선 129", "머선129", "존버", "짤", "개이득", "소확행", "꾸안꾸", "먹방",
    "썸", "갑툭튀", "반사", "싫존주의", "식집사", "깊꾸", "신꾸", "폴꾸", "캘박",
    "웃안웃", "완내스", "좋댓구알", "많관부", "어쩔티비", "쫌쫌따리", "군싹",
    "억텐", "찐텐", "애빼시", "복세편살", "자만추", "인만추", "오저치고", "불소킹",
    "전공", "코시국", "구취", "딱내스", "안내스", "당모치", "돈쭐", "핑프",
    "혼밥", "혼술", "혼영", "졌잘싸", "너또다", "최최차차", "라떼는 말이야",
    "레게노", "뇌피셜", "뉴트로", "딸바보", "랜선생님", "위쑤시개", "한풀루언서",
    "무지컬", "손절미", "테무인간", "밥플릭스", "수발새끼", "윗치다꺼리",
    "진지스칸", "쇼믈리에", "릴셉션", "고민세", "요들갑", "위고빔", "배타인지",
    "내원내", "니원내", "선원내", "부원내",

    # 게임/인터넷 / 커뮤
    "지지", "GG", "뇌절", "사바사", "스불재", "갑분싸", "문찐", "팬아저", "꾸꾸꾸",
    "현웃", "솔까말", "남아공", "멘붕", "행쇼", "느좋", "사이다", "고구마",

    # 초성/야민정음/의성어
    "야민정음", "초성체", "ㅇㅈ", "ㅋㅋ", "ㅎㅎ", "ㄴㅇㄱ",

    # 오타/밈 계열
    "고나리", "쵸재깅", "샨새교", "애냑마", "줗녀",

    # 영어 혼합 계열
    "고트", "G.O.A.T", "홀리몰리", "홀리몰리과카몰리",

    # 인물/관계/외모 밈
    "훈남", "완소남", "딸친아", "얼죽아", "중꺾마", "이왜진", "킹리적갓심",
    "제곧내", "탕진잼", "투더", "뻘하다", "뽀짝", "뿌잉뿌잉", "초딩", "잼민이",
    "엄친아", "엄친딸",

    # 네가 추가한 합성어/신조어들
    "합만하앗", "무지", "팀듐", "스브재", "증껍마", "시강", "좀좀마리", "어사",
    "랜선", "랜선수업",  # 랜선~ 패턴도 꽤 있어서 같이 넣음

    # 다른 자주 쓰이는 것들 보강
    "오졌다", "지렸다", "찐이다", "ㅇㅈ이지", "레전드다", "미쳤다", "미쳤어", "ㅎㅂㅎ",
    "알빠노", "노알라", "노답", "현타", "현타 온다", "심쿵", "존맛탱", "존잼",
    "퇴사각", "재질", "국룰", "극락", "댕청", "갑분띠", "소름돋네", "국프",
    "꼰대", "혼코노", "랜선모임", "멍때리기", "혼코노", "스겜", "빡겜",

    # 기타 Z세대/밈 표현 보강
    "쩐다", "갓벽", "떡상", "폭망", "오열", "현생", "스리슬쩍", "꿀템",
    "갬성", "감성샷", "감성폭발", "인생샷", "인생곡", "인생영화", "힐링타임",

    # 위쪽에서 빠졌을 수 있는 것들 중 다시 한 번
    "쿠쿠루삥뽕", "랜선생", "위쑤시개", "밥플릭스"
]

# 중복 제거(혹시 모를 중복 대비)
SLANG_KEYWORDS = sorted(set(SLANG_KEYWORDS))

# 세대 / 감정 라벨 후보
GENERATIONS = ["10s", "20s", "30s", "40s+"]
EMOTIONS = ["긍정", "부정", "중립"]

# 생성할 데이터 수
NUM_DATA = 10000  # 처음에는 500~2000으로 돌려보고, 괜찮으면 늘려가는 걸 추천

# -----------------------------------------------------------------------------
# 2. SYSTEM 프롬프트
#    - GPT가 source/target/generation/emotion 을 한 번에 만들어줌
# -----------------------------------------------------------------------------
SLANG_LIST_FOR_PROMPT = ", ".join(SLANG_KEYWORDS)

SYSTEM_PROMPT = f"""
너는 한국어 신조어·은어, 특히 10~20대가 SNS(유튜브 쇼츠, 틱톡, 릴스, 인스타그램 등)에서 
많이 사용하는 표현을 매우 잘 이해하는 언어 전문가다.

### 임무

1) 아래 조건을 만족하는 '신조어 문장(source)'을 1개 생성해라.
   - 아래 슬랭 리스트 중 **적어도 1개 이상**을 반드시 포함해야 한다.
   - 슬랭은 자연스럽게 섞어서 사용한다. (무리하게 여러 개를 넣지 말 것)
   - 실제 사람들이 쓸 법한 대화/상황 문장으로 만든다.
   - 지나치게 과장되거나 무의미한 반복(ㅋㅋㅋㅋㅋㅋㅋㅋ, 의미 없는 자모 반복 등)은 피한다.
   - 욕설/비하 표현은 '이미 널리 쓰이는 밈/인터넷 표현' 범위 내에서만 사용하고, 불필요하게 과도한 혐오 표현은 피한다.

   슬랭 리스트(예시):
   {SLANG_LIST_FOR_PROMPT}

2) 위 문장을 '다른 세대(30~40대 이상)도 이해할 수 있는 자연스러운 표준어 문장(target)'으로 변환해라.
   - 의미는 최대한 그대로 유지한다.
   - 감정(긍정/부정/중립) 뉘앙스를 유지한다.
   - 새로운 정보를 임의로 추가하지 않는다.

3) 문장의 화자 관점과 상황을 고려해 아래 항목도 함께 지정하라.
   - generation: ["10s", "20s", "30s", "40s+"] 중 하나
   - emotion: ["긍정", "부정", "중립"] 중 하나

### 출력 형식 (JSON only)

설명, 코드블록, 여는 말/닫는 말 없이 **오직 아래 JSON 한 개만** 출력한다.

{{
  "source": "<신조어 문장 한 문장>",
  "target": "<표준어/쉬운 표현 한 문장>",
  "generation": "<10s|20s|30s|40s+ 중 하나>",
  "emotion": "<긍정|부정|중립 중 하나>"
}}
"""

# -----------------------------------------------------------------------------
# 3. 샘플 후처리 & 품질 필터
# -----------------------------------------------------------------------------


def contains_slang(text: str) -> bool:
    """슬랭 키워드 중 1개 이상 포함되는지 체크"""
    return any(kw in text for kw in SLANG_KEYWORDS)


def is_valid_text(text: str) -> bool:
    """너무 짧거나 이상한 텍스트 필터링"""
    text = text.strip()

    # 길이 제한
    if not (6 <= len(text) <= 120):
        return False

    # 한글이 하나도 없으면 제외
    if not re.search(r"[가-힣]", text):
        return False

    # 같은 문자 5회 이상 반복(ex. ㅋㅋㅋㅋㅋㅋㅋㅋ)
    if re.search(r"(.)\1{5,}", text):
        return False

    # 특수문자 비율이 너무 높으면 제거
    num_special = len(re.findall(r"[^0-9A-Za-z가-힣\s]", text))
    if num_special / max(len(text), 1) > 0.4:
        return False

    return True


# -----------------------------------------------------------------------------
# 4. GPT 한 번 호출해서 샘플 생성
# -----------------------------------------------------------------------------


def generate_sample(max_retry=3):
    for _ in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "한 쌍의 신조어 문장과 변환 문장을 생성해줘."},
                ],
            )

            content = resp.choices[0].message.content
            data = json.loads(content)

            source = data.get("source", "").strip()
            target = data.get("target", "").strip()
            generation = data.get("generation", "").strip()
            emotion = data.get("emotion", "").strip()

            # 기본 필터
            if not source or not target:
                return None

            if not is_valid_text(source) or not is_valid_text(target):
                return None

            # 슬랭이 1개 이상 포함되어 있는지 체크
            if not contains_slang(source):
                return None

            # generation / emotion 값 이상하면 랜덤 보정
            if generation not in GENERATIONS:
                generation = random.choice(GENERATIONS)
            if emotion not in EMOTIONS:
                emotion = random.choice(EMOTIONS)

            return {
                "generation": generation,
                "emotion": emotion,
                "source": source,
                "target": target,
            }

        except Exception as e:
            print("API 오류, 재시도 중:", e)
            time.sleep(1)

    return None


# -----------------------------------------------------------------------------
# 5. CSV로 저장
# -----------------------------------------------------------------------------


def main():
    output_path = f"slang_dataset_{NUM_DATA}.csv"

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "generation", "emotion","source", "target"])

        idx = 1
        seen_sources = set()

        for _ in tqdm(range(NUM_DATA), desc="Generating"):
            sample = None
            # 품질 필터에 계속 걸리면 몇 번 재시도
            for _ in range(8):
                sample = generate_sample()
                if sample is None:
                    continue
                # source 중복 제거
                if sample["source"] in seen_sources:
                    sample = None
                    continue
                break

            if sample is None:
                # 이 iteration은 스킵
                continue

            seen_sources.add(sample["source"])
            sample_id = f"{idx:05d}"

            writer.writerow([
                sample_id,
                sample["generation"],
                sample["emotion"],
                sample["source"],
                sample["target"],
            ])

            idx += 1

            # 레이트 리밋 방지용 (필요하면 조절)
            time.sleep(0.05)

    print(f"{output_path} 생성 완료!  (총 {idx - 1}개 샘플)")


if __name__ == "__main__":
    main()
