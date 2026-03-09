#%%
# ============================================
# Multi-Horizon Prediction Module
# 1d / 3d / 5d
# Ridge + LGBM Ensemble
# ============================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# -------------------------------------------------
# 1️⃣ 데이터 로드 + 타겟 생성
# -------------------------------------------------
def load_and_prepare(path):

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 3일 / 5일 누적 로그수익률
    df["target_3d"] = df["log_return_t+1"].shift(-2).rolling(3).sum()
    df["target_5d"] = df["log_return_t+1"].shift(-4).rolling(5).sum()

    return df.dropna().reset_index(drop=True)


# -------------------------------------------------
# 2️⃣ Feature 선택
# -------------------------------------------------
def select_features(df):

    features = []

    for i in range(1, 8):
        features.append(f"sub_index{i}_lag1")
        features.append(f"sub_index{i}_lag2")

    features += [
        "sent_norm_w",
        "sent_energy",
        "sent_std_inv",
        "neg_z_inv",
        "sent_norm_ma5",
        "neg_z_ma5",
        "KFGI",
        "dayofweek",
        "month",
    ]

    return features


# -------------------------------------------------
# 3️⃣ 예측 함수 (Ridge + LGBM 앙상블)
# -------------------------------------------------
def predict_horizon(df, features, target_col):

    X = df[features]
    y = df[target_col]
    y_cls = (y > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    out = []

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_train_cls = y_cls.iloc[train_idx]

        # -------------------------
        # Ridge (with scaling)
        # -------------------------
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        ridge_pred = ridge.predict(X_test_s)

        # -------------------------
        # LGBM
        # -------------------------
        lgbm = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgb_pred = lgbm.predict(X_test)

        # -------------------------
        # Ensemble (단순 평균)
        # -------------------------
        pred_ret = 0.5 * ridge_pred + 0.5 * lgb_pred

        # 분류 확률 (LGBM만 사용)
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_train, y_train_cls)
        pred_prob = clf.predict_proba(X_test)[:, 1]

        tmp = pd.DataFrame({
            "date": df.loc[test_idx, "date"].values,
            "actual": y_test.values,
            "pred_ret": pred_ret,
            "pred_prob_up": pred_prob,
        })

        out.append(tmp)

    return pd.concat(out).sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# 4️⃣ 실행부
# -------------------------------------------------
if __name__ == "__main__":

    df = load_and_prepare("KFG_with_KFGI.csv")
    features = select_features(df)

    print("Running 1-day prediction...")
    pred_1d = predict_horizon(df, features, "log_return_t+1")
    pred_1d = pred_1d.rename(columns={
        "pred_ret": "pred_ret_1d",
        "pred_prob_up": "pred_prob_up_1d"
    })

    print("Running 3-day prediction...")
    pred_3d = predict_horizon(df, features, "target_3d")
    pred_3d = pred_3d.rename(columns={
        "pred_ret": "pred_ret_3d",
        "pred_prob_up": "pred_prob_up_3d"
    })

    print("Running 5-day prediction...")
    pred_5d = predict_horizon(df, features, "target_5d")
    pred_5d = pred_5d.rename(columns={
        "pred_ret": "pred_ret_5d",
        "pred_prob_up": "pred_prob_up_5d"
    })

    # 날짜 기준 병합
    pred_all = pred_1d.merge(
        pred_3d[["date", "pred_ret_3d", "pred_prob_up_3d"]],
        on="date",
        how="left"
    ).merge(
        pred_5d[["date", "pred_ret_5d", "pred_prob_up_5d"]],
        on="date",
        how="left"
    )

    pred_all.to_csv("prediction_multihorizon.csv", index=False)

    print("Saved: prediction_multihorizon.csv")


#%%
# ===============================
# 방향성 분석용 df 로드 (중요)
# ===============================
pred_df = pd.read_csv("prediction_multihorizon.csv")
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df = pred_df.sort_values("date").reset_index(drop=True)

orig_df = pd.read_csv("KFG_with_KFGI.csv")
orig_df["date"] = pd.to_datetime(orig_df["date"])

# 실제 1일 수익률 merge
pred_df = pred_df.merge(
    orig_df[["date", "log_return_t+1"]],
    on="date",
    how="left"
).rename(columns={"log_return_t+1": "actual_1d"})

pred_df = pred_df.dropna().reset_index(drop=True)

# ===============================
# 신호 정의 (여기서부터 안전)
# ===============================
pred_df["signal_1d"] = (pred_df["pred_ret_1d"] > 0).astype(int)
pred_df["signal_3d"] = (pred_df["pred_ret_3d"] > 0).astype(int)
pred_df["signal_5d"] = (pred_df["pred_ret_5d"] > 0).astype(int)

th = pred_df["pred_prob_up_1d"].quantile(0.8)
pred_df["signal_top20"] = (pred_df["pred_prob_up_1d"] >= th).astype(int)

# ===============================
# 방향성 평가 실행
# ===============================
direction_analysis(pred_df["signal_1d"], pred_df["actual_1d"], "1D Strategy")
direction_analysis(pred_df["signal_3d"], pred_df["actual_1d"], "3D Strategy")
direction_analysis(pred_df["signal_5d"], pred_df["actual_1d"], "5D Strategy")
direction_analysis(pred_df["signal_top20"], pred_df["actual_1d"], "Top20 Strategy")





#%%
# 검증
'''
이 코드 돌렸을 때, 얻을 수 잇는 건 예측값이 0보다 클 때 매수 전략이 5일 후 수익률에 있어서 큰 성능을 보임
-> 1일 수익률 예측에 있어서는 노이즈로 인해, 실제 많은 연구에서도 난항을 겪으나
5일 후 수익률에 있어서는 노이즈 완화로 인해 괜찮은 듯 함.
'''
import pandas as pd
import numpy as np

df = pd.read_csv("prediction_multihorizon.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 실제 1일 수익률 필요
original = pd.read_csv("KFG_with_KFGI.csv")
original["date"] = pd.to_datetime(original["date"])

df = df.merge(
    original[["date", "log_return_t+1"]],
    on="date",
    how="left"
)

df.rename(columns={"log_return_t+1": "actual_1d"}, inplace=True)

# ===============================
# 전략 1: pred_ret > 0 이면 진입
# ===============================
df["signal_1d"] = (df["pred_ret_1d"] > 0).astype(int)
df["strategy_ret_1d"] = df["signal_1d"] * df["actual_1d"]
df["strategy_ret_3d"] = (df["pred_ret_3d"] > 0).astype(int) * df["actual_1d"]
df["strategy_ret_5d"] = (df["pred_ret_5d"] > 0).astype(int) * df["actual_1d"]
# ===============================
# 전략 2: 상위 20% 확률만 진입
# ===============================
th = df["pred_prob_up_1d"].quantile(0.8)
df["signal_top20"] = (df["pred_prob_up_1d"] >= th).astype(int)
df["strategy_ret_top20"] = df["signal_top20"] * df["actual_1d"]

# ===============================
# 성능 함수
# ===============================
def performance(ret, name):

    ret = ret.dropna()

    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    cum = np.exp(ret.cumsum())
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    mdd = drawdown.min()

    print(f"\n===== {name} =====")
    print(f"Annual Return: {ann_ret:.3f}")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Max Drawdown: {mdd:.3f}")

# ===============================
# 실행
# ===============================
performance(df["strategy_ret_1d"], "Pred > 0 Strategy")
performance(df["strategy_ret_3d"], "Pred > 0 Strategy 3")
performance(df["strategy_ret_5d"], "Pred > 0 Strategy 5")
performance(df["strategy_ret_top20"], "Top 20% Prob Strategy")
performance(df["actual_1d"], "Buy & Hold")





#%%
'''
이 코드 돌려보면 1일 후 수익률에 있어서는 감성점수를 넣는 게 솔직히 아무 의미 없지만
5일 후 예측에 있어서는 성능 차이가 존재(positive)
'''
# ===============================================
# Sentiment 효과 비교 실험 (최종 정리본)
# ===============================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit


# -------------------------------------------------
# 1️⃣ Feature 생성
# -------------------------------------------------
def build_features(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df.ffill().bfill()

    # ----- 감성 파생 -----
    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]

    df["sent_norm_ma5"] = df["sent_norm_w"].rolling(5).mean()
    df["neg_z_ma5"] = df["neg_z"].rolling(5).mean()

    # ----- Sub-index lag -----
    for i in range(1, 8):
        df[f"sub_index{i}_lag1"] = df[f"sub_index{i}"].shift(1)

    # ----- 캘린더 -----
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ----- 타겟 -----
    df["target_reg"] = df["log_return_t+1"]
    df["target_reg_5d"] = df["log_return_t+1"].shift(-4).rolling(5).sum()

    return df.dropna().reset_index(drop=True)


# -------------------------------------------------
# 2️⃣ KFGI (Sent 포함)
# -------------------------------------------------
def create_kfgi_sent(df, train_end="2024-12-31"):

    df = df.copy()
    train_mask = df["date"] <= train_end

    core_feats = [f"sub_index{i}_lag1" for i in range(1, 8)] + [
        "sent_norm_w",
        "sent_energy",
        "sent_std_inv",
        "neg_z_inv",
    ]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all = scaler.transform(df[core_feats])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train, df.loc[train_mask, "target_reg"])

    w = ridge.coef_
    w = w / (np.sum(np.abs(w)) + 1e-12)

    raw = X_all @ w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])

    scaled = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)

    df["K_FGI"] = scaled
    df["KFGI"] = 100 - scaled

    return df


# -------------------------------------------------
# 3️⃣ KFGI (Sent 미포함)
# -------------------------------------------------
def create_kfgi_no_sent(df, train_end="2024-12-31"):

    df = df.copy()
    train_mask = df["date"] <= train_end

    core_feats = [f"sub_index{i}_lag1" for i in range(1, 8)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all = scaler.transform(df[core_feats])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train, df.loc[train_mask, "target_reg"])

    w = ridge.coef_
    w = w / (np.sum(np.abs(w)) + 1e-12)

    raw = X_all @ w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])

    scaled = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)

    df["K_FGI_no_sent"] = scaled
    df["KFGI_no_sent"] = 100 - scaled

    return df


# -------------------------------------------------
# 4️⃣ Sent 효과 비교
# -------------------------------------------------
def compare_sent_effect(df, target_col):

    tscv = TimeSeriesSplit(n_splits=5)

    base_feats = [f"sub_index{i}_lag1" for i in range(1, 8)] + ["dayofweek", "month"]

    feats_A = base_feats + ["KFGI_no_sent"]
    feats_B = base_feats + ["KFGI"]
    feats_C = feats_B + ["sent_norm_w", "sent_energy", "neg_z_inv"]

    results = {}

    for name, feats in {
        "No Sent KFGI": feats_A,
        "Sent KFGI": feats_B,
        "Sent + Direct": feats_C
    }.items():

        preds, actual = [], []

        X = df[feats]
        y = df[target_col]

        for train_idx, test_idx in tscv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)

            pred = model.predict(X_test_s)

            preds.extend(pred)
            actual.extend(y_test)

        preds = np.array(preds)
        actual = np.array(actual)

        strategy = (preds > 0) * actual

        ann_ret = strategy.mean() * 252
        ann_vol = strategy.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)

        corr = np.corrcoef(preds, actual)[0, 1]

        results[name] = (round(sharpe, 3), round(corr, 3))

    return results


# -------------------------------------------------
# 5️⃣ 실행부
# -------------------------------------------------
input_path = "/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv"

df_raw = pd.read_csv(input_path)

df_feat = build_features(df_raw)
df_feat = create_kfgi_sent(df_feat)
df_feat = create_kfgi_no_sent(df_feat)

results_1d = compare_sent_effect(df_feat, "target_reg")
results_5d = compare_sent_effect(df_feat, "target_reg_5d")

print("===== 1D 결과 =====")
print(results_1d)

print("\n===== 5D 결과 =====")
print(results_5d)


#%%
#%%
# ============================================
# Strategy Evaluation & Visualization
# prediction_multihorizon.csv 기준
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 데이터 로드
# ----------------------------
df_pred = pd.read_csv("prediction_multihorizon.csv")
df_pred["date"] = pd.to_datetime(df_pred["date"])

df_orig = pd.read_csv("KFG_with_KFGI.csv")
df_orig["date"] = pd.to_datetime(df_orig["date"])

df = df_pred.merge(
    df_orig[["date", "log_return_t+1", "KFGI"]],
    on="date",
    how="left"
)

df = df.sort_values("date").reset_index(drop=True)
df.rename(columns={"log_return_t+1": "actual_1d"}, inplace=True)

# ----------------------------
# 전략 정의
# ----------------------------

# 1D
df["ret_1d"] = (df["pred_ret_1d"] > 0).astype(int) * df["actual_1d"]

# 3D 예측 → 1일 수익률에 적용
df["ret_3d"] = (df["pred_ret_3d"] > 0).astype(int) * df["actual_1d"]

# 5D 예측 → 1일 수익률에 적용
df["ret_5d"] = (df["pred_ret_5d"] > 0).astype(int) * df["actual_1d"]

# 상위 20% 확률 전략
th = df["pred_prob_up_1d"].quantile(0.8)
df["ret_top20"] = (df["pred_prob_up_1d"] >= th).astype(int) * df["actual_1d"]


# ----------------------------
# 성과 함수
# ----------------------------
def performance(log_ret):
    ann_ret = log_ret.mean() * 252
    ann_vol = log_ret.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    cum = np.exp(log_ret.cumsum())
    mdd = (cum / cum.cummax() - 1).min()

    return ann_ret, sharpe, mdd


# ----------------------------
# 시각화 함수
# ----------------------------
def plot_strategy(strat_col, name):

    ann_ret_m, sharpe_m, mdd_m = performance(df["actual_1d"])
    ann_ret_s, sharpe_s, mdd_s = performance(df[strat_col])

    df["market_cum"] = np.exp(df["actual_1d"].cumsum())
    df["strat_cum"] = np.exp(df[strat_col].cumsum())

    plt.figure(figsize=(12,6))
    plt.plot(df["date"], df["market_cum"], label=f"Market (S:{sharpe_m:.2f})", color="gray", linewidth=2)
    plt.plot(df["date"], df["strat_cum"], label=f"{name} (S:{sharpe_s:.2f})", color="red", linewidth=2)

    plt.title(f"{name} vs Market")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n===== Performance =====")
    print("Market  :", {"Return": f"{ann_ret_m:.2%}", "Sharpe": round(sharpe_m,2), "MDD": round(mdd_m,2)})
    print(name, ":", {"Return": f"{ann_ret_s:.2%}", "Sharpe": round(sharpe_s,2), "MDD": round(mdd_s,2)})


# ----------------------------
# 실행
# ----------------------------

plot_strategy("ret_1d", "Pred > 0 (1D)")
plot_strategy("ret_3d", "Pred > 0 (3D)")
plot_strategy("ret_5d", "Pred > 0 (5D)")
plot_strategy("ret_top20", "Top 20% Prob Strategy")


#%%
#%%
# KFGI 포함 시각화

def plot_with_kfgi(strat_col, name):

    # --- 성과 계산 ---
    ann_ret_m, sharpe_m, mdd_m = performance(df["actual_1d"])
    ann_ret_s, sharpe_s, mdd_s = performance(df[strat_col])

    df["market_cum"] = np.exp(df["actual_1d"].cumsum())
    df["strat_cum"] = np.exp(df[strat_col].cumsum())

    # --- 그래프 ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12,8), sharex=True,
        gridspec_kw={"height_ratios": [3,1]}
    )

    # 1️⃣ 위: 누적수익
    ax1.plot(df["date"], df["market_cum"],
             label=f"Market (S:{sharpe_m:.2f})",
             color="black", linewidth=2)

    ax1.plot(df["date"], df["strat_cum"],
             label=f"{name} (S:{sharpe_s:.2f})",
             color="red", linewidth=2)

    ax1.set_title(f"{name} vs Market")
    ax1.legend()
    ax1.grid(True)

    # 2️⃣ 아래: KFGI
    ax2.plot(df["date"], df["KFGI"],
             color="blue", linewidth=1)

    ax2.axhline(df["KFGI"].quantile(0.2),
                linestyle="--", color="gray", alpha=0.6)

    ax2.axhline(df["KFGI"].quantile(0.8),
                linestyle="--", color="gray", alpha=0.6)

    ax2.set_ylabel("KFGI")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n===== Performance =====")
    print("Market  :", {"Return": f"{ann_ret_m:.2%}",
                         "Sharpe": round(sharpe_m,2),
                         "MDD": round(mdd_m,2)})
    print(name, ":", {"Return": f"{ann_ret_s:.2%}",
                      "Sharpe": round(sharpe_s,2),
                      "MDD": round(mdd_s,2)})

plot_with_kfgi("ret_5d", "Pred > 0 (5D)")

#%%
def plot_with_kfgi_regime(strat_col, name):

    df_plot = df.copy().dropna().reset_index(drop=True)

    # 분위수 기준
    fear_th = df_plot["KFGI"].quantile(0.2)
    greed_th = df_plot["KFGI"].quantile(0.8)

    # 누적수익
    df_plot["market_cum"] = np.exp(df_plot["actual_1d"].cumsum())
    df_plot["strat_cum"] = np.exp(df_plot[strat_col].cumsum())

    fig, ax = plt.subplots(figsize=(14,6))

    # -------------------------
    # 배경 음영 (Regime 구분)
    # -------------------------
    for i in range(len(df_plot)-1):
        if df_plot["KFGI"].iloc[i] <= fear_th:
            ax.axvspan(df_plot["date"].iloc[i],
                       df_plot["date"].iloc[i+1],
                       color="blue", alpha=0.08)
        elif df_plot["KFGI"].iloc[i] >= greed_th:
            ax.axvspan(df_plot["date"].iloc[i],
                       df_plot["date"].iloc[i+1],
                       color="red", alpha=0.08)

    # -------------------------
    # 수익곡선
    # -------------------------
    ax.plot(df_plot["date"], df_plot["market_cum"],
            label="Market", color="black", linewidth=2)

    ax.plot(df_plot["date"], df_plot["strat_cum"],
            label=name, color="green", linewidth=2)

    ax.set_title(f"{name} vs Market (Fear/Greed Regime Highlighted)")
    ax.legend()
    ax.grid(True)

    plt.show()

    print(f"Fear Threshold: {round(fear_th,2)}")
    print(f"Greed Threshold: {round(greed_th,2)}")

plot_with_kfgi_regime("ret_5d", "Pred > 0 (5D)")
