#%%
# ============================================
# KFGI Pipeline
# 1) Feature Engineering
# 2) KFGI Construction (Ridge-based)
# ============================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV


# -------------------------------------------------
# 1️⃣ Feature Engineering
# -------------------------------------------------
def build_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 결측 보간
    df = df.ffill().bfill()

    # ---- 감성 파생 변수 ----
    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]

    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_diff"] = df["neg_z"].diff()

    df["sent_norm_ma5"] = df["sent_norm_w"].rolling(5).mean()
    df["neg_z_ma5"] = df["neg_z"].rolling(5).mean()

    # ---- Sub-index lag ----
    sub_cols = [f"sub_index{i}" for i in range(1, 8)]
    for col in sub_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    # ---- 캘린더 변수 ----
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ---- 타겟 ----
    df["target_reg"] = df["log_return_t+1"]
    df["target_cls"] = (df["log_return_t+1"] > 0).astype(int)

    return df.dropna().reset_index(drop=True)


# -------------------------------------------------
# 2️⃣ KFGI Construction
# -------------------------------------------------
def create_kfgi(df, train_end="2024-12-31"):
    df = df.copy()
    train_mask = df["date"] <= train_end

    # 핵심 입력 변수 (7 sub + 감성 4개)
    core_feats = [f"sub_index{i}" for i in range(1, 8)] + [
        "sent_norm_w",
        "sent_energy",
        "sent_std_inv",
        "neg_z_inv",
    ]

    # 스케일링 (train 기준)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all = scaler.transform(df[core_feats])

    # Ridge로 가중치 학습
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train, df.loc[train_mask, "target_reg"])

    w = ridge.coef_
    w = w / (np.sum(np.abs(w)) + 1e-12)

    # Raw index
    raw_score = X_all @ w

    # Train 구간 기준 1~99% clipping
    p1, p99 = np.percentile(raw_score[train_mask], [1, 99])
    scaled = 100 * (np.clip(raw_score, p1, p99) - p1) / (p99 - p1 + 1e-12)

    df["K_FGI"] = scaled
    df["KFGI"] = 100 - df["K_FGI"]

    return df, w


# -------------------------------------------------
# 3️⃣ Main Execution
# -------------------------------------------------
if __name__ == "__main__":

    input_path = '/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv'  # 경로 맞게 수정
    df_raw = pd.read_csv(input_path)

    # 1) Feature Engineering
    df_feat = build_features(df_raw)

    # 2) KFGI 생성
    df_kfgi, weights = create_kfgi(df_feat)

    # 결과 저장
    df_kfgi.to_csv("KFG_with_KFGI.csv", index=False)

    print("KFGI 생성 완료")
    print("가중치:")
    print(weights)






