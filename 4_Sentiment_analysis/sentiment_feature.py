import pandas as pd
from glob import glob
import numpy as np

'''
연도별 감성 점수 파일을 로드하여 하나의 데이터프레임으로 병합
'''
def load_year(year):
    pattern = f"data/NAVER/sentiment_scores/sentiment_with_prob_{year}_*.csv"
    files = glob(pattern)

    if len(files) == 0:
        raise ValueError(f"{year} 파일이 없습니다. 패턴: {pattern}")

    print(f"{year} → {len(files)}개 파일 로드")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["comment_at"] = pd.to_datetime(df["comment_at"], utc=True)
    df["comment_at"] = df["comment_at"].dt.tz_convert("Asia/Seoul")
    df["date"] = df["comment_at"].dt.date

    return df

df_2022 = load_year(2022)
df_2023 = load_year(2023)
df_2024 = load_year(2024)
df_2025 = load_year(2025)

print("2022:", len(df_2022))
print("2023:", len(df_2023))
print("2024:", len(df_2024))
print("2025:", len(df_2025))


# 모든 연도 병합
df_all = pd.concat([df_2022, df_2023, df_2024, df_2025])
df_all = df_all.sort_values("date").reset_index(drop=True)


'''
감성 지표 계산 및 일별 집계
- sent_norm_w: 방향성 (긍정/부정 균형)
- sent_strength_w: 확신 (감정화 정도)
- sent_std: 불확실성 (감정의 일관성)
- neg_z: 충격 변수 (부정 감정의 비정상적 급증)
- effective_n: 참여도 (가중치 기반 유효 댓글 수)
- heat: 감정 강도와 참여도의 복합 지표
'''
def make_daily(df):

    df = df.copy()

    # -----------------------------
    # 1. 댓글 단위 감성 계산
    # -----------------------------
    df["sent_raw"] = df["p_pos"] - df["p_neg"]

    df["sent_raw_weighted"] = df["sent_raw"] * df["weight"]
    df["p_pos_weighted"] = df["p_pos"] * df["weight"]
    df["p_neg_weighted"] = df["p_neg"] * df["weight"]

    grouped = df.groupby("date")

    # -----------------------------
    # 2. 일별 집계
    # -----------------------------
    daily = grouped.agg(
        weight_sum=("weight", "sum"),
        weight_sq_sum=("weight", lambda x: (x**2).sum()),
        sent_std=("sent_raw", "std"),
    )

    # Weighted mean 확률
    daily["pos_mean_w"] = (
        grouped["p_pos_weighted"].sum() / daily["weight_sum"]
    )

    daily["neg_mean_w"] = (
        grouped["p_neg_weighted"].sum() / daily["weight_sum"]
    )

    # -----------------------------
    # 3. 핵심 지표 변수
    # -----------------------------

    # 방향성
    daily["sent_norm_w"] = (
        (daily["pos_mean_w"] - daily["neg_mean_w"]) /
        (daily["pos_mean_w"] + daily["neg_mean_w"] + 1e-8)
    )

    # 확신 (중립 제외 감정화 정도)
    daily["sent_strength_w"] = (
        daily["pos_mean_w"] + daily["neg_mean_w"]
    )

    # 불확실성은 이미 sent_std

    # -----------------------------
    # 4. 충격 변수 (neg_z)
    # -----------------------------
    daily = daily.sort_index()

    daily["neg_score"] = daily["neg_mean_w"]

    roll_mean_60 = daily["neg_score"].rolling(60).mean()
    roll_std_60 = daily["neg_score"].rolling(60).std()

    daily["neg_z"] = (
        (daily["neg_score"] - roll_mean_60) /
        (roll_std_60 + 1e-8)
    )

    # -----------------------------
    # 5. 참여도 (보조용)
    # -----------------------------
    daily["effective_n"] = (
        (daily["weight_sum"]**2) /
        (daily["weight_sq_sum"] + 1e-8)
    )

    daily["log_effective_n"] = np.log1p(daily["effective_n"])

    # -----------------------------
    # 6. Heat (보조 이벤트 지표)
    # -----------------------------
    # Z-score 함수
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    daily["heat"] = (
        zscore(daily["sent_strength_w"]) *
        zscore(daily["log_effective_n"])
    )

    # -----------------------------
    # 최종 반환 컬럼만 정리
    # -----------------------------
    daily = daily.reset_index()

    final_cols = [
        "date",
        "sent_norm_w",
        "sent_strength_w",
        "sent_std",
        "neg_z",

        "effective_n",
        "heat",
    ]

    return daily[final_cols]


daily_all = make_daily(df_all)
daily_all = daily_all.dropna().reset_index(drop=True)

import os

output_dir = "data/NAVER/sentiment_final"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/daily_sentiment.csv"

daily_all.to_csv(output_path, index=False)
print("일별 감성 요약 저장 완료: data/NAVER/sentiment_final/daily_sentiment_summary.csv")


