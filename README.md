# 🚀 Generation-tailored Slang Filtering & Translation System
### 세대 맞춤형 신조어·은어 자동 번역 모델 (KcT5 Fine-tuning)

한국어 신조어·은어를 문맥 기반으로 탐지하고, 세대별 이해도에 맞게 자연스러운 문장으로 변환하는 NLP 프로젝트입니다.

---

## 📚 Overview

SNS·커뮤니티에서 빠르게 변화하는 신조어·은어는 세대 간 의사소통의 장벽이 되고 있습니다.  
본 프로젝트는 **KcT5-small Transformer 모델(KcT5-base Transformer모델도 구현)을 파인튜닝하여** 다음 기능을 수행합니다:

- 🧭 신조어·은어 자동 탐지  
- 🔄 세대 맞춤형 표준어 변환  
- 😊 감정(긍정/부정/중립) 유지 변환  
- 🛡 공격적·부적절 표현 자동 필터링  
- 🧠 문맥 기반 자연스러운 생성  

---

## 📁 Dataset Structure

CSV 데이터는 다음 필드로 구성됩니다:

| Column | Description |
|--------|-------------|
| id | 고유 ID |
| generation | 세대 구분 (10s, 20s 등) |
| emotion | 감정 (긍정/부정/중립) |
| source | 신조어 문장 |
| target | 표준어 문장 |

### 📌 Example
generation: 10s
emotion: 긍정
source: 오늘 알잘딱깔센하게 준비했어.
target: 오늘 알아서 깔끔하고 센스 있게 준비했어.


---

## 🧩 Model

- Base Model: **KcT5-small**
- 선택 이유:
  - 데이터 규모(약 1200개)에 비해 base 모델이 과대 → gradient explosion 발생  
  - small 모델로 변경 후 학습 안정성 확보  

---

## ⚙️ Training Details

### 🔥 1. Gradient Explosion 해결
- 초기에 grad_norm이 1000 이상 튀는 문제 발생  
- **→ Gradient Clipping 적용**하여 안정화  

### 🌀 2. Model Collapse 문제
초기 학습에서 반복 발생:
- 무의미한 토큰 반복  
- WSJ/NYT 스타일 문장 생성  
- Prefix 과의존  

**해결 방법**
- warmup scheduler 적용  
- learning rate 초기값 조정  
- prefix를 자연어 기반으로 단순화  

### 🧪 3. 데이터 분포 불균형 해결
- handmade 데이터 편중  
- ai-generated 문장 추가하여 길이·구조 다양성 확보  

---

## 🎯 Contributions

- 한국어 신조어 → 표준어 변환 모델을 위한 **안정적 파인튜닝 프로토콜 최초 제안**  
- 소규모 Transformer에서 발견되는 **gradient explosion 원인을 규명**  
- prefix 구조가 학습 안정성에 미치는 영향 분석  
- 자연어 기반 prompt가 성능을 안정화함을 실증  
- warmup + clipping이 model collapse를 해소함을 확인  

---

## 📂 Project Structure
```
NLP-Slang-Translator/
│
├── data/
│   └── slang_dataset.csv
│
├── models/
│   └── kcT5-small/
│
├── train.py
├── slang_generator.py
├── requirements.txt
└── README.md
```
