#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns


df_fgi = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_index_analysis.csv')
# 3. 통합 비교 실험 및 피처 중요도 분석
def run_full_analysis(df, train_end='2024-12-31'):
    train_mask = df['date'] <= train_end
    test_mask = df['date'] > train_end

    sub_only = [f'sub_index{i}_lag1' for i in range(1, 8)] + ['dayofweek', 'month']
    with_sent_ridge = sub_only + ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv',
                            'sent_norm_diff', 'neg_z_diff', 'sent_norm_ma5', 'neg_z_ma5', 'K_FGI_Ridge']
    with_sent_pca = sub_only + ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv',
                            'sent_norm_diff', 'neg_z_diff', 'sent_norm_ma5', 'neg_z_ma5', 'K_FGI_PCA']
    with_sent_fa = sub_only + ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv',
                            'sent_norm_diff', 'neg_z_diff', 'sent_norm_ma5', 'neg_z_ma5', 'K_FGI_FA']  

    experiments = [
        ('Reg', 'Sub Only', sub_only),
        ('Reg', 'Sub+Sent_ridge', with_sent_ridge),
        ('Reg', 'Sub+Sent_pca', with_sent_pca),
        ('Reg', 'Sub+Sent_fa', with_sent_fa),
        ('Cls', 'Sub Only', sub_only),
        ('Cls', 'Sub+Sent_ridge', with_sent_ridge),
        ('Cls', 'Sub+Sent_pca', with_sent_pca),
        ('Cls', 'Sub+Sent_fa', with_sent_fa)
    ]

    summary = []
    fig_cum, ax_cum = plt.subplots(figsize=(10, 5))
    fig_imp, axes_imp = plt.subplots(2, 4, figsize=(15, 12))
    axes_imp = axes_imp.flatten()

    for i, (m_type, f_type, f_list) in enumerate(experiments):
        target = 'target_reg' if m_type == 'Reg' else 'target_cls'

        # 모델 학습
        dtrain = lgb.Dataset(df.loc[train_mask, f_list], label=df.loc[train_mask, target],
                             weight=df.loc[train_mask, 'sample_weight'])
        params = {'objective': 'regression' if m_type == 'Reg' else 'binary', 'verbosity': -1, 'learning_rate': 0.02}
        model = lgb.train(params, dtrain, num_boost_round=300)

        # 예측 및 성과
        preds = model.predict(df.loc[test_mask, f_list])
        actual_ret = df.loc[test_mask, 'log_return_t+1']
        signal = (preds > 0) if m_type == 'Reg' else (preds > 0.5)
        strat_ret = signal * actual_ret

        # 지표 산출
        ann_ret = strat_ret.mean() * 252
        ann_vol = (strat_ret.std() * np.sqrt(252)) + 1e-9
        sharpe = ann_ret / ann_vol

        summary.append({'Model': f"{m_type}_{f_type}", 'Sharpe': round(sharpe, 3), 'Return': f"{ann_ret*100:.2f}%"})

        # 수익 곡선 플롯
        ax_cum.plot(df.loc[test_mask, 'date'], np.exp(strat_ret.cumsum()), label=f"{m_type}_{f_type} (S:{round(sharpe,2)})")

        # 피처 중요도 플롯
        importances = pd.DataFrame({'Feature': f_list, 'Importance': model.feature_importance(importance_type='gain')})
        importances = importances.sort_values(by='Importance', ascending=False).head(10)
        sns.barplot(x='Importance', y='Feature', data=importances, ax=axes_imp[i], palette='viridis')
        axes_imp[i].set_title(f"Top 10 Features: {m_type}_{f_type}")

    # 최종 결과 정리
    ax_cum.plot(df.loc[test_mask, 'date'], np.exp(actual_ret.cumsum()), 'k--', label='Market', alpha=0.3)
    ax_cum.set_title("Cumulative Returns Comparison")
    ax_cum.legend(); ax_cum.grid(True)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary)

# 실행 및 결과 출력
results = run_full_analysis(df_fgi)
print("\n===== 벤치마크 결과 요약 =====")
print(results.to_string(index=False))

#%%
#%%
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA, FactorAnalysis

def prep_df(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def make_indices_trainfit(df, train_mask,
                          ridge_feats=None, pca_feats=None, fa_feats=None,
                          ridge_target_col="target_reg",
                          clip_pct=(1, 99)):
    """
    df: 전체 데이터(정렬됨)
    train_mask: boolean mask (train 구간)
    반환: df에 K_FGI_Ridge/K_FGI_PCA/K_FGI_FA 컬럼을 생성한 복사본
    """
    df2 = df.copy()

    if ridge_feats is None:
        ridge_feats = [f"sub_index{i}" for i in range(1, 8)] + ["sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv"]
    if pca_feats is None:
        pca_feats = ridge_feats
    if fa_feats is None:
        fa_feats = ridge_feats

    # ---- Ridge Index (지도지만: fit은 train만) ----
    sc_r = StandardScaler()
    Xr_tr = sc_r.fit_transform(df2.loc[train_mask, ridge_feats])
    Xr_all = sc_r.transform(df2[ridge_feats])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Xr_tr, df2.loc[train_mask, ridge_target_col])
    w = ridge.coef_
    w = w / (np.sum(np.abs(w)) + 1e-12)
    raw = Xr_all @ w

    p1, p99 = np.percentile(raw[train_mask], clip_pct)
    df2["K_FGI_Ridge"] = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)

    # ---- PCA Index (비지도) ----
    sc_p = StandardScaler()
    Xp_tr = sc_p.fit_transform(df2.loc[train_mask, pca_feats])
    Xp_all = sc_p.transform(df2[pca_feats])

    pca = PCA(n_components=1)
    pca.fit(Xp_tr)
    pc1 = pca.transform(Xp_all).ravel()

    # 방향 고정(선택): neg_z(공포)가 클수록 지수는 낮게(=탐욕↑) 만들고 싶으면
    # 현재 df에는 neg_z_inv가 있으니, neg_z_inv와 +상관이면 그대로 두는 편이 자연스러움.
    # 그래도 혹시 뒤집고 싶으면 아래 주석 해제:
    # corr = np.corrcoef(pc1[train_mask], df2.loc[train_mask, "neg_z_inv"])[0, 1]
    # if corr < 0: pc1 = -pc1

    p1, p99 = np.percentile(pc1[train_mask], clip_pct)
    df2["K_FGI_PCA"] = 100 * (np.clip(pc1, p1, p99) - p1) / (p99 - p1 + 1e-12)

    # ---- FA Index (비지도) ----
    sc_f = StandardScaler()
    Xf_tr = sc_f.fit_transform(df2.loc[train_mask, fa_feats])
    Xf_all = sc_f.transform(df2[fa_feats])

    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(Xf_tr)
    f1 = fa.transform(Xf_all).ravel()

    # 방향 고정(원하면): neg_z_inv와 양의 상관이 되게
    # corr = np.corrcoef(f1[train_mask], df2.loc[train_mask, "neg_z_inv"])[0, 1]
    # if corr < 0: f1 = -f1

    p1, p99 = np.percentile(f1[train_mask], clip_pct)
    df2["K_FGI_FA"] = 100 * (np.clip(f1, p1, p99) - p1) / (p99 - p1 + 1e-12)

    return df2

def lgb_train_predict(df, train_mask, test_mask, features, target_col, sample_weight_col="sample_weight",
                      m_type="Reg"):
    dtrain = lgb.Dataset(
        df.loc[train_mask, features],
        label=df.loc[train_mask, target_col],
        weight=df.loc[train_mask, sample_weight_col] if sample_weight_col in df.columns else None
    )
    params = {
        "objective": "regression" if m_type == "Reg" else "binary",
        "verbosity": -1,
        "learning_rate": 0.02,
        # 과적합 완화(기본 안전장치 몇 개)
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
    }
    model = lgb.train(params, dtrain, num_boost_round=400)
    preds = model.predict(df.loc[test_mask, features])
    return preds

def walk_forward_eval(
    df,
    start_train="2023-01-01",
    first_test="2024-01-01",
    test_months=3,
    step_months=3,
    rolling_train_months=None,  # None이면 expanding, 숫자(예:24)면 rolling 24개월
):
    """
    df에는 아래가 있어야 함:
    - date
    - log_return_t+1
    - target_reg, target_cls (있으면)
    - sample_weight (있으면)

    반환:
    - fold별 결과 DataFrame
    - 전체 test 구간의 일별 전략수익률(모델별)
    """
    df = prep_df(df)

    df["date"] = pd.to_datetime(df["date"])
    start_train = pd.to_datetime(start_train)
    first_test = pd.to_datetime(first_test)

    # Feature sets
    sub_only = [f"sub_index{i}_lag1" for i in range(1, 8)] + ["dayofweek", "month"]

    with_sent_ridge = sub_only + [
        "sent_norm_w","sent_energy","sent_std_inv","neg_z_inv",
        "sent_norm_diff","neg_z_diff","sent_norm_ma5","neg_z_ma5","K_FGI_Ridge"
    ]
    with_sent_pca = sub_only + [
        "sent_norm_w","sent_energy","sent_std_inv","neg_z_inv",
        "sent_norm_diff","neg_z_diff","sent_norm_ma5","neg_z_ma5","K_FGI_PCA"
    ]
    with_sent_fa = sub_only + [
        "sent_norm_w","sent_energy","sent_std_inv","neg_z_inv",
        "sent_norm_diff","neg_z_diff","sent_norm_ma5","neg_z_ma5","K_FGI_FA"
    ]

    experiments = [
        ("Reg", "Sub Only", sub_only),
        ("Reg", "Sub+Sent_ridge", with_sent_ridge),
        ("Reg", "Sub+Sent_pca", with_sent_pca),
        ("Reg", "Sub+Sent_fa", with_sent_fa),
        ("Cls", "Sub Only", sub_only),
        ("Cls", "Sub+Sent_ridge", with_sent_ridge),
        ("Cls", "Sub+Sent_pca", with_sent_pca),
        ("Cls", "Sub+Sent_fa", with_sent_fa),
    ]

    # fold 루프
    fold_rows = []
    # 모델별 일별 수익률 누적 저장
    daily_returns = {f"{m}_{name}": [] for (m, name, _) in experiments}
    daily_dates = []

    test_start = first_test
    while True:
        test_end = test_start + pd.DateOffset(months=test_months)
        test_mask = (df["date"] >= test_start) & (df["date"] < test_end)
        if test_mask.sum() == 0:
            break

        # train 구간 설정
        if rolling_train_months is None:
            train_start = start_train
        else:
            train_start = test_start - pd.DateOffset(months=rolling_train_months)

        train_mask = (df["date"] >= train_start) & (df["date"] < test_start)

        # 학습 데이터가 너무 적으면 중단
        if train_mask.sum() < 100:
            break

        # ✅ fold별 지수 재생성 (train-fit)
        df_fold = make_indices_trainfit(df, train_mask)

        # fold의 날짜 저장(일별 수익률 병합용)
        fold_dates = df_fold.loc[test_mask, "date"].tolist()
        if len(daily_dates) == 0:
            daily_dates = fold_dates
        else:
            daily_dates += fold_dates

        # 실제 수익률
        actual_ret = df_fold.loc[test_mask, "log_return_t+1"].values

        for (m_type, name, f_list) in experiments:
            target = "target_reg" if m_type == "Reg" else "target_cls"
            preds = lgb_train_predict(df_fold, train_mask, test_mask, f_list, target, m_type=m_type)

            signal = (preds > 0) if m_type == "Reg" else (preds > 0.5)
            strat_ret = signal.astype(int) * actual_ret

            ann_ret = strat_ret.mean() * 252
            ann_vol = strat_ret.std() * np.sqrt(252) + 1e-12
            sharpe = ann_ret / ann_vol

            fold_rows.append({
                "fold_train_start": train_start.date(),
                "fold_test_start": test_start.date(),
                "fold_test_end": (test_end - pd.Timedelta(days=1)).date(),
                "model": f"{m_type}_{name}",
                "test_days": int(test_mask.sum()),
                "ann_ret": ann_ret,
                "sharpe": sharpe
            })

            daily_returns[f"{m_type}_{name}"] += strat_ret.tolist()

        # 다음 fold로 이동
        test_start = test_start + pd.DateOffset(months=step_months)

    folds_df = pd.DataFrame(fold_rows)

    # 전체기간 성과(모델별) 요약
    summary_rows = []
    for k, rets in daily_returns.items():
        rets = np.array(rets, dtype=float)
        if len(rets) == 0:
            continue
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252) + 1e-12
        sharpe = ann_ret / ann_vol
        cum = np.exp(np.cumsum(rets))[-1]
        summary_rows.append({
            "model": k,
            "ann_ret(%)": round(ann_ret * 100, 2),
            "sharpe": round(sharpe, 3),
            "final_cum": round(float(cum), 2),
            "n_days": int(len(rets))
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)

    # 일별 수익률 데이터프레임(원하면 그래프)
    daily_df = pd.DataFrame({"date": daily_dates})
    for k, rets in daily_returns.items():
        daily_df[k] = rets[:len(daily_df)]

    return folds_df, summary_df, daily_df

#%%
df_fgi = prep_df(df_fgi)
df_fgi["date"] = pd.to_datetime(df_fgi["date"])

folds_df, summary_df, daily_df = walk_forward_eval(
    df_fgi,
    start_train="2023-01-01",
    first_test="2024-01-01",
    test_months=3,
    step_months=3,
    rolling_train_months=None  # expanding
)

print("\n===== Fold별 결과(일부) =====")
print(folds_df.head(10).to_string(index=False))

print("\n===== 전체 Walk-forward 요약 =====")
print(summary_df.to_string(index=False))

#%%
folds_df_r, summary_df_r, daily_df_r = walk_forward_eval(
    df_fgi,
    start_train="2023-01-01",
    first_test="2024-01-01",
    test_months=3,
    step_months=3,
    rolling_train_months=24
)
print(summary_df_r.to_string(index=False))

#%%
