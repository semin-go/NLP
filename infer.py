from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./kcslang-model"

# 모델 / 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)


# 테스트용 단순 함수
def translate_sentence(sentence):
    if isinstance(sentence, list):
        return [translate_sentence(s) for s in sentence]

    input_text = "표준어로 바꿔줘: " + sentence

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    )

    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    src = [
        "오늘 알잘딱깔센하게 놀아보자 했어.",
        "진짜 킹받네.",
        "오늘 하루 개뿌듯하다.",
        "와 이 카페 갬성 미쳤다.",
    ]

    print("원문:", src)
    
    results = translate_sentence(src)

    print("\n번역 결과:")
    for s, t in zip(src, results):
        print(f"- {s} → {t}")
