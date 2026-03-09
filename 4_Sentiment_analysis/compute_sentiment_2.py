'''
python geonho/sentiment_analysis/compute_sentiment_2.py \
  --year 2023 \
  --name geonho

python geonho/sentiment_analysis/compute_sentiment_2.py \
  --year 2024 \
  --name geonho

python geonho/sentiment_analysis/compute_sentiment_2.py \
  --year 2025 \
  --name geonho

각각 터미널에서 실행, 코드, 파일 경로는 바꿔야  함
'''
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ==========================
# 1. Argument
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

year = args.year
name = args.name

file_path = f"data/NAVER/final_filtered/comments_final_stock_only_{year}.csv"
save_dir = f"data/NAVER/sentiment_scores"
os.makedirs(save_dir, exist_ok=True)

print(f"===== {year} 시작 =====")

# ==========================
# 2. Load
# ==========================

df = pd.read_csv(file_path)

df["comment_at"] = pd.to_datetime(df["comment_at"])
df = df[df["comment_at"].dt.year == year]

df = df[(df["is_empty"] == 0) & (df["keep"] == 1)]
df = df.dropna(subset=["text_raw"])
df = df[df["text_raw"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print("댓글 수:", len(df))

# ==========================
# 3. 모델 로드
# ==========================

model_name = "snunlp/KR-FinBert-SC"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

neg_id = model.config.label2id["negative"]
neu_id = model.config.label2id["neutral"]
pos_id = model.config.label2id["positive"]

# ==========================
# 4. 확률 추출 함수
# ==========================

@torch.no_grad()
def get_probabilities(texts, batch_size=64, max_length=64):

    all_pos = []
    all_neu = []
    all_neg = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)

        all_pos.extend(probs[:, pos_id].cpu().numpy())
        all_neu.extend(probs[:, neu_id].cpu().numpy())
        all_neg.extend(probs[:, neg_id].cpu().numpy())

    return all_pos, all_neu, all_neg


print("확률 계산 시작")
p_pos, p_neu, p_neg = get_probabilities(df["text_raw"].tolist())
print("확률 계산 완료")

# ==========================
# 5. 컬럼 추가
# ==========================

df["p_pos"] = p_pos
df["p_neu"] = p_neu
df["p_neg"] = p_neg

# (선택) 기존 sentiment_score도 같이 남기고 싶으면
df["sentiment_raw"] = df["p_pos"] - df["p_neg"]

# ==========================
# 6. 저장
# ==========================

output_path = f"{save_dir}/sentiment_with_prob_{year}_{name}.csv"
df.to_csv(output_path, index=False)

print(f"{year} 저장 완료 -> {output_path}")