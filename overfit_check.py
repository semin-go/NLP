import os, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)

HF_CACHE_DIR = r"C:\hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

MODEL_NAME = "KETI-AIR/ke-t5-small"
DATA_PATH = "slang_dataset_10000.csv"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
SEED = 42

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_function(examples, tokenizer):
    inputs = ["표준어: " + s for s in examples["source"]]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # ✅ 권장 방식
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    # ✅ pad는 loss에서 무시하도록 -100으로 (안정적으로)
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in lab]
        for lab in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

def generate_one(model, tokenizer, device, src: str):
    inp = tokenizer("표준어: " + src, return_tensors="pt", truncation=True, max_length=128).to(device)
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

def main():
    set_seed(SEED)

    raw = load_dataset("csv", data_files={"data": DATA_PATH}, encoding="utf-8-sig")["data"]
    raw = raw.shuffle(seed=SEED)

    # ✅ 일부만 떼서 “학습이 되는지” 확인
    train_ds = raw.select(range(100))
    valid_ds = raw.select(range(100, 120))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    tokenized_train = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True,
                                   remove_columns=train_ds.column_names)
    tokenized_valid = valid_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True,
                                   remove_columns=valid_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    args = TrainingArguments(
        output_dir="./overfit-ckpt",
        overwrite_output_dir=True,
        num_train_epochs=30,            # ✅ 오버피팅 목적
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        weight_decay=0.0,
        logging_steps=5,
        save_steps=10**9,               # ✅ 사실상 저장 안 하게
        fp16=False,                     # ✅ 일단 fp16 끄고 안정성 확인
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("🚀 오버피팅 학습 시작!")
    trainer.train()

    # ✅ 학습 데이터 5개로 생성 확인
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("\n==============================")
    print("✅ TRAIN OVERFIT GENERATION CHECK")
    print("==============================")

    samples = train_ds.select(range(5))
    for ex in samples:
        src = ex["source"]
        tgt = ex["target"]
        pred = generate_one(model, tokenizer, device, src)
        print("\nSRC:", src)
        print("TGT:", tgt)
        print("PRD:", pred)

if __name__ == "__main__":
    main()
