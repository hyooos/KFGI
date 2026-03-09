#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 가이드 기반 피처 엔지니어링 (동일)
df = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True).fillna(method='ffill').fillna(method='bfill')

def build_advanced_features(df):
    df = df.copy()
    df['neg_z_inv'] = -df['neg_z']
    df['sent_std_inv'] = -df['sent_std']
    df['sent_energy'] = df['sent_strength_w'] * df['sent_norm_w']
    df['sent_norm_diff'] = df['sent_norm_w'].diff()
    df['neg_z_diff'] = df['neg_z'].diff()
    df['sent_norm_ma5'] = df['sent_norm_w'].rolling(5).mean()
    df['neg_z_ma5'] = df['neg_z'].rolling(5).mean()

    sub_cols = [f'sub_index{i}' for i in range(1, 8)]
    for col in sub_cols:
        df[f'{col}_lag1'] = df[col].shift(1)

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['target_reg'] = df['log_return_t+1']
    df['target_cls'] = (df['log_return_t+1'] > 0).astype(int)
    df['sample_weight'] = np.log1p(df['effective_n'])

    return df.dropna().reset_index(drop=True)

df_eng = build_advanced_features(df)


def create_kfgi_pca(df, train_end='2024-12-31', fear_proxy_col=None):
    """
    PCA 기반 K-FGI 생성
    - 비지도 방식
    - train 구간 기준으로 PCA 적합
    - 방향 자동 정렬 (탐욕 ↑)
    - 0~100 스케일링
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 1️⃣ 사용할 핵심 변수 정의
    core_feats = [f'sub_index{i}' for i in range(1, 8)] + \
                 ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv']

    # 2️⃣ train/test 마스크
    train_mask = df['date'] <= train_end

    # 3️⃣ 스케일링 (train 기준)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all_scaled = scaler.transform(df[core_feats])

    # 4️⃣ PCA (1번째 주성분 사용)
    pca = PCA(n_components=1)
    pca.fit(X_train_scaled)

    pc1_all = pca.transform(X_all_scaled).flatten()
    pc1_train = pc1_all[train_mask]

    # 5️⃣ 방향 자동 정렬 (선택)
    # fear_proxy_col이 주어지면, 공포와 음의 상관이 되도록 부호 조정
    if fear_proxy_col is not None:
        corr = np.corrcoef(pc1_train, df.loc[train_mask, fear_proxy_col])[0,1]
        if corr > 0:
            pc1_all = -pc1_all
            pc1_train = -pc1_train

    # 6️⃣ train 분포 기준 0~100 스케일링
    p1, p99 = np.percentile(pc1_train, [1, 99])
    kfgi = 100 * (np.clip(pc1_all, p1, p99) - p1) / (p99 - p1)

    df['K_FGI_PCA'] = kfgi

    print("\n===== PCA 설명력 =====")
    print(f"Explained Variance Ratio (PC1): {pca.explained_variance_ratio_[0]:.3f}")

    print("\n===== PCA 가중치 =====")
    weights = pd.Series(pca.components_[0], index=core_feats)
    print(weights.sort_values(ascending=False))

    return df, pca, scaler, weights

df_pca, pca_model, scaler_model, pca_weights = create_kfgi_pca(
    df_eng,
    train_end='2024-12-31',
    fear_proxy_col='neg_z'  # 선택사항 (없으면 None)
)
#%%
plt.figure(figsize=(10,5))
plt.plot(df_pca['date'], df_pca['K_FGI_PCA'])
plt.title("K-FGI (PCA Based)")
plt.ylim(0,100)
plt.grid(True)
plt.show()

#%%
rolling_std = df_pca['K_FGI_PCA'].rolling(60).std()
df_pca[['K_FGI_PCA', 'log_return_t+1']].corr()

#%%
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

def create_kfgi_fa(df, train_end='2024-12-31', n_factors=1, fear_proxy_col=None):
    """
    Factor Analysis 기반 K-FGI 생성
    - 로그수익률 사용 안 함
    - train 구간 기준 학습
    - 방향 자동 정렬
    - 0~100 스케일링
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    core_feats = [f'sub_index{i}' for i in range(1, 8)] + \
                 ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv']

    train_mask = df['date'] <= train_end

    # 1️⃣ 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all_scaled = scaler.transform(df[core_feats])

    # 2️⃣ Factor Analysis
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(X_train_scaled)

    factor_all = fa.transform(X_all_scaled)
    factor_train = factor_all[train_mask]

    # 3️⃣ 1요인일 경우 flatten
    if n_factors == 1:
        factor_all = factor_all.flatten()
        factor_train = factor_train.flatten()

        # 4️⃣ 방향 자동 정렬
        if fear_proxy_col is not None:
            corr = np.corrcoef(factor_train, df.loc[train_mask, fear_proxy_col])[0,1]
            if corr > 0:
                factor_all = -factor_all
                factor_train = -factor_train

        # 5️⃣ 스케일링
        p1, p99 = np.percentile(factor_train, [1, 99])
        df['K_FGI_FA'] = 100 * (np.clip(factor_all, p1, p99) - p1) / (p99 - p1)

    else:
        # 다요인일 경우 첫 번째 요인만 사용
        factor_all_1 = factor_all[:,0]
        factor_train_1 = factor_train[:,0]

        if fear_proxy_col is not None:
            corr = np.corrcoef(factor_train_1, df.loc[train_mask, fear_proxy_col])[0,1]
            if corr > 0:
                factor_all_1 = -factor_all_1
                factor_train_1 = -factor_train_1

        p1, p99 = np.percentile(factor_train_1, [1, 99])
        df['K_FGI_FA'] = 100 * (np.clip(factor_all_1, p1, p99) - p1) / (p99 - p1)

    # 6️⃣ 요인 적재량 출력
    loadings = pd.Series(fa.components_[0], index=core_feats)

    print("\n===== Factor Loadings =====")
    print(loadings.sort_values(ascending=False))

    return df, fa, scaler, loadings

df_fa, fa_model, scaler_model, loadings = create_kfgi_fa(
    df_eng,
    train_end='2024-12-31',
    n_factors=1,
    fear_proxy_col='neg_z'  # 선택사항
)

#%%
plt.figure(figsize=(10,5))
plt.plot(df_fa['date'], df_fa['K_FGI_FA'])
plt.title("K-FGI (Factor Analysis)")
plt.ylim(0,100)
plt.grid(True)
plt.show()


#%%


# 2. K-FGI 지수 생성 (Ridge)
def create_kfgi(df, train_end='2024-12-31'):
    train_mask = df['date'] <= train_end
    core_feats = [f'sub_index{i}' for i in range(1, 8)] + \
                 ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all_scaled = scaler.transform(df[core_feats])
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_train_scaled, df.loc[train_mask, 'target_reg'])
    w = ridge.coef_ / np.sum(np.abs(ridge.coef_))
    raw = X_all_scaled @ w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])
    df['K_FGI'] = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1)
    return df, core_feats

df_fgi, core_feats = create_kfgi(df_eng)
#%%
fgi_pca = df_pca['K_FGI_PCA']
fgi_fa = df_fa['K_FGI_FA']
fgi_ridge = df_fgi['K_FGI']

df_final = pd.merge(df, fgi_pca, fgi_fa, fgi_ridge)
df_final.to_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_index_analysis.csv')


#%%
df_final = df_eng.copy()

df_final['K_FGI_PCA'] = df_pca['K_FGI_PCA'].values
df_final['K_FGI_FA'] = df_fa['K_FGI_FA'].values
df_final['K_FGI_Ridge'] = df_fgi['K_FGI'].values

df_final.to_csv(
    '/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_index_analysis.csv',
    index=False
)

