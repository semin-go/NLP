import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./kcslang-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

print("토크나이저 경로:", tokenizer.name_or_path)
print("모델 경로:", model.name_or_path)
print("eos_token_id:", tokenizer.eos_token_id, "pad_token_id:", tokenizer.pad_token_id)

def translate_slang(sentence: str) -> str:
    input_text = "표준어로 바꿔줘: " + sentence

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **enc,
            # ✅ 길이 제어: max_length 대신 max_new_tokens 추천
            max_new_tokens=64,

            # ✅ 반복 붕괴 방지
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,

            # ✅ 종료 토큰 명시(중요)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    tests = [
        "오늘 알잘딱깔센하게 준비했어",
        "진짜 킹받네",
        "하루 종일 현타온다",
        "오늘 룩 완전 갓벽이야",
    ]
    for t in tests:
        print(f"입력: {t}")
        print(f"출력: {translate_slang(t)}")
        print("-" * 40)
