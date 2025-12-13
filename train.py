import os
import shutil
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# =========================
# 0. 캐시 설정
# =========================
HF_CACHE_DIR = r"C:\hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# =========================
# 1. 기본 설정
# =========================
MODEL_NAME = "KETI-AIR/ke-t5-small"
DATA_PATH = "slang_dataset_10000.csv"

OUTPUT_DIR = "./kcslang-stable-ckpt"
SAVE_DIR = "./kcslang-stable-model"

MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64
SEED = 42

# ✅ 오버핏 단계: 20개로 먼저 성공시킨 뒤 100→1000→전체
TRAIN_SIZE = 5000
VALID_SIZE = 2000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 2. 데이터 로드
# =========================
def load_small_split(csv_path: str, train_size: int, valid_size: int):
    raw = load_dataset(
        "csv",
        data_files={"data": csv_path},
        encoding="utf-8-sig",
    )["data"]

    raw = raw.shuffle(seed=SEED)

    total_need = train_size + valid_size
    if len(raw) < total_need:
        raise ValueError(f"데이터 개수가 부족함: 현재 {len(raw)}개, 최소 {total_need}개 필요")

    train_ds = raw.select(range(train_size))
    valid_ds = raw.select(range(train_size, train_size + valid_size))

    print("✅ 데이터 로드 완료")
    print("  - 전체:", len(raw))
    print("  - train:", len(train_ds))
    print("  - valid:", len(valid_ds))
    print("  - columns:", raw.column_names)
    return train_ds, valid_ds


# =========================
# 3. 전처리 (중요: </s> 같은 EOS 토큰 넣지 않기)
# =========================
def preprocess_function(examples, tokenizer):
    inputs = [f"은어: {s}\n표준어:" for s in examples["source"]]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,  # ✅ collator가 동적패딩 처리
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs


def translate(model, tokenizer, text: str):
    model.eval()
    inp = tokenizer(
        f"은어: {text}\n표준어:",
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# =========================
# 4. 메인
# =========================
def main():
    set_seed(SEED)

    # ✅ (필수) 예전 체크포인트가 있으면 학습이 망가진 상태로 이어질 수 있음
    # 오버핏 성공 확인 전까진 무조건 새로 시작하자
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

    train_dataset, valid_dataset = load_small_split(DATA_PATH, TRAIN_SIZE, VALID_SIZE)

    print("✅ 토크나이저/모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(DEVICE)

    # ✅ (필수) 학습 전 sanity check: 라벨이 정상인지 확인
    ex = train_dataset[0]
    print("\n[SANITY CHECK]")
    print("SRC:", ex["source"])
    print("TGT:", ex["target"])
    lab_ids = tokenizer(text_target=ex["target"], truncation=True, max_length=MAX_TARGET_LENGTH)["input_ids"]
    print("label_len:", len(lab_ids))
    print("label_decoded:", tokenizer.decode(lab_ids, skip_special_tokens=True))

    print("\n✅ 전처리 중...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_valid = valid_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    # ✅ 동적패딩 + label pad는 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,

        # ✅ 오버핏은 오래 돌려서 붙는지 확인
        num_train_epochs=50,

        # ✅ 오버핏에서는 acc를 크게 잡지 말자 (불안정해짐)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # ✅ 16 → 4로 감소

        # ✅ T5 안정화
        adafactor=True,
        learning_rate=1e-6,
        warmup_ratio=0.1,
        max_grad_norm=0.2,

        weight_decay=0.0,
        logging_steps=2,
        save_steps=10**9,  # ✅ 오버핏 단계에서는 저장 거의 안함
        fp16=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("🚀 학습 시작! (resume_from_checkpoint=False)")
    trainer.train(resume_from_checkpoint=False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ 모델 저장 완료: {SAVE_DIR}")

    print("\n==============================")
    print("✅ QUICK GENERATION TEST (TRAIN SAMPLES)")
    print("==============================")
    samples = train_dataset.select(range(min(5, len(train_dataset))))
    for ex in samples:
        src = ex["source"]
        tgt = ex["target"]
        pred = translate(model, tokenizer, src)
        print("\nSRC:", src)
        print("TGT:", tgt)
        print("PRD:", pred)


if __name__ == "__main__":
    main()
