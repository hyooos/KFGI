#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
import shap

# --- Step 1 & 3: 피처 엔지니어링 (사용자님 logic + 고도화) ---
def build_advanced_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True).fillna(method='ffill').fillna(method='bfill')

    # 감성 에너지 및 모멘텀
    df['sent_energy'] = df['sent_strength_w'] * df['sent_norm_w']
    df['sent_norm_ma5'] = df['sent_norm_w'].rolling(5).mean()
    df['neg_z_inv'] = -df['neg_z']

    # 멀티 Lag 생성 (1일, 2일 전 정보)
    for i in range(1, 8):
        df[f'sub_index{i}_lag1'] = df[f'sub_index{i}'].shift(1)

    # 타겟 생성 (1일 수익률)
    df['target'] = df['log_return_t+1']
    df['target_cls'] = (df['target'] > 0).astype(int)

    return df.dropna().reset_index(drop=True)

# --- Step 5: K-FGI 생성 (경제적 직관 반영) ---
def create_logical_kfgi(df):
    core_feats = [f'sub_index{i}' for i in range(1, 8)] + ['sent_norm_w', 'sent_energy', 'neg_z_inv']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[core_feats])

    # 수익률과의 상관성으로 가중치 산출하되, 절대값 사용하여 중요도만 추출
    ridge = RidgeCV().fit(X_scaled, df['target'])
    weights = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))

    # 0~100 스케일링 (지수가 높을수록 탐욕)
    raw_fgi = X_scaled @ weights
    p1, p99 = np.percentile(raw_fgi, [1, 99])
    df['KFGI'] = 100 * (np.clip(raw_fgi, p1, p99) - p1) / (p99 - p1)
    return df, core_feats

def create_logical_kfgi_no_sent(df):
    core_feats = [f'sub_index{i}' for i in range(1, 8)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[core_feats])

    # 수익률과의 상관성으로 가중치 산출하되, 절대값 사용하여 중요도만 추출
    ridge = RidgeCV().fit(X_scaled, df['target'])
    weights = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))

    # 0~100 스케일링 (지수가 높을수록 탐욕)
    raw_fgi = X_scaled @ weights
    p1, p99 = np.percentile(raw_fgi, [1, 99])
    df['KFGI'] = 100 * (np.clip(raw_fgi, p1, p99) - p1) / (p99 - p1)
    return df, core_feats
#%%
# 데이터 준비
df_raw = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv')
df_eng = build_advanced_features(df_raw)
df_final, core_feats = create_logical_kfgi(df_eng)
df_final_no, core_feats_no = create_logical_kfgi_no_sent(df_eng)

#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 초기화 (사용자님의 파일 경로에 맞춰주세요)
# df_raw가 이미 있다면 그대로 사용하고, 없다면 새로 읽어옵니다.
# df_raw = KFG_final_2
# KFGI 생성

try:
    df = df_final.copy()
except NameError:
    df = pd.read_csv('KFG_final_2.csv')

# 2. 타겟 및 피처 자동 식별
# 타겟 변수 설정 (사용자님 코드의 'actual_1d' 혹은 'log_return_t+1' 자동 선택)
target_col = 'log_return_t+1' if 'log_return_t+1' in df.columns else 'actual_1d'
if target_col not in df.columns:
    # 만약 둘 다 없다면 마지막 컬럼을 타겟으로 가정 (위험 방지용)
    target_col = df.columns[-1]

# 피처 리스트 자동 생성 (날짜, 타겟, 종가 등을 제외한 모든 숫자형 컬럼)
exclude_cols = ['date', 'kospi_close', 'log_return', 'log_return_t+1', 'actual_1d', 'target_reg', 'target_3d', 'target_5d', 'KFGI', 'target','target_cls']
features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

print(f"✅ 사용 타겟: {target_col}")
print(f"✅ 사용 피처 ({len(features)}개): {features}")

# 3. 시계열 교차 검증 및 하이브리드 예측 함수
def run_final_strategy(df, features, target):
    tscv = TimeSeriesSplit(n_splits=5)
    X = df[features].fillna(0)
    y = df[target]

    results = []
    val_errors = []

    print("🔄 모델 학습 및 시계열 검증 진행 중...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Ridge + LGBM 앙상블 (수익성 위주)
        model_r = Ridge(alpha=1.0).fit(X_train, y_train)
        model_l = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=63, verbosity=-1)
        model_l.fit(X_train, y_train)

        # 앙상블 예측 (비선형 패턴에 80% 비중)
        pred = (model_r.predict(X_test) * 0.2) + (model_l.predict(X_test) * 0.8)

        # 과적합 체크를 위한 오차 저장
        val_errors.append(np.sqrt(np.mean((y_test - pred)**2)))

        tmp = pd.DataFrame({
            'date': pd.to_datetime(df.iloc[test_idx]['date']),
            'actual': y_test,
            'pred': pred,
            'kfgi': df.iloc[test_idx]['KFGI'] if 'KFGI' in df.columns else 50 # KFGI 없으면 기본값
        })
        results.append(tmp)

    return pd.concat(results), val_errors

# 4. 전략 실행
res_df, errors = run_final_strategy(df, features, target_col)

# 5. 🔥 수익 극대화 로직 (시장을 이기기 위한 공격적 세팅)
# 상승장 소외를 막기 위해 아주 약간의 상승 가능성만 있어도 매수
res_df['signal'] = np.where(res_df['pred'] > -0.0002, 1, 0)

# KFGI 기반 동적 비중 (공포에 사고 탐욕에 줄이기)
res_df['weight'] = 1.0
res_df.loc[res_df['kfgi'] < 30, 'weight'] = 1.8  # 과공포 시 1.8배 공격적 매수
res_df.loc[res_df['kfgi'] > 80, 'weight'] = 0.6  # 과탐욕 시 비중 축소

# 최종 전략 수익률
res_df['strat_ret'] = res_df['signal'] * res_df['weight'] * res_df['actual']

# 6. 성과 리포트 및 과적합 진단
print("\n" + "="*40)
print("🏆 최종 전략 성과 보고서")
print("="*40)


#%%
features_no = ['sub_index1',
 'sub_index2',
 'sub_index3',
 'sub_index4',
 'sub_index5',
 'sub_index6',
 'sub_index7',
 'sub_index1_lag1',
 'sub_index2_lag1',
 'sub_index3_lag1',
 'sub_index4_lag1',
 'sub_index5_lag1',
 'sub_index6_lag1',
 'sub_index7_lag1']

try:
    df = df_final_no.copy()
except NameError:
    df = pd.read_csv('KFG_final_2.csv')

# 2. 타겟 및 피처 자동 식별
# 타겟 변수 설정 (사용자님 코드의 'actual_1d' 혹은 'log_return_t+1' 자동 선택)
target_col = 'log_return_t+1' if 'log_return_t+1' in df.columns else 'actual_1d'
if target_col not in df.columns:
    # 만약 둘 다 없다면 마지막 컬럼을 타겟으로 가정 (위험 방지용)
    target_col = df.columns[-1]

# 피처 리스트 자동 생성 (날짜, 타겟, 종가 등을 제외한 모든 숫자형 컬럼)
exclude_cols = ['date', 'kospi_close', 'log_return', 'log_return_t+1', 'actual_1d', 'target_reg', 'target_3d', 'target_5d', 'KFGI', 'target','target_cls']
features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

print(f"✅ 사용 타겟: {target_col}")
print(f"✅ 사용 피처 ({len(features_no)}개): {features_no}")

# 3. 시계열 교차 검증 및 하이브리드 예측 함수
def run_final_strategy(df, features, target):
    tscv = TimeSeriesSplit(n_splits=5)
    X = df[features].fillna(0)
    y = df[target]

    results = []
    val_errors = []

    print("🔄 모델 학습 및 시계열 검증 진행 중...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Ridge + LGBM 앙상블 (수익성 위주)
        model_r = Ridge(alpha=1.0).fit(X_train, y_train)
        model_l = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=63, verbosity=-1)
        model_l.fit(X_train, y_train)

        # 앙상블 예측 (비선형 패턴에 80% 비중)
        pred = (model_r.predict(X_test) * 0.2) + (model_l.predict(X_test) * 0.8)

        # 과적합 체크를 위한 오차 저장
        val_errors.append(np.sqrt(np.mean((y_test - pred)**2)))

        tmp = pd.DataFrame({
            'date': pd.to_datetime(df.iloc[test_idx]['date']),
            'actual': y_test,
            'pred': pred,
            'kfgi': df.iloc[test_idx]['KFGI'] if 'KFGI' in df.columns else 50 # KFGI 없으면 기본값
        })
        results.append(tmp)

    return pd.concat(results), val_errors

# 4. 전략 실행
res_df_no, errors = run_final_strategy(df, features_no, target_col)


#%%
# 5. 🔥 수익 극대화 로직 (시장을 이기기 위한 공격적 세팅)
# 상승장 소외를 막기 위해 아주 약간의 상승 가능성만 있어도 매수
res_df_no['signal'] = np.where(res_df_no['pred'] > -0.0002, 1, 0)

# KFGI 기반 동적 비중 (공포에 사고 탐욕에 줄이기)
res_df_no['weight'] = 1.0
res_df_no.loc[res_df_no['kfgi'] < 30, 'weight'] = 1.8  # 과공포 시 1.8배 공격적 매수
res_df_no.loc[res_df_no['kfgi'] > 80, 'weight'] = 0.6  # 과탐욕 시 비중 축소

# 최종 전략 수익률
res_df_no['strat_ret'] = res_df_no['signal'] * res_df_no['weight'] * res_df_no['actual']

# 6. 성과 리포트 및 과적합 진단
print("\n" + "="*40)
print("🏆 최종 전략 성과 보고서")
print("="*40)

# %%
#%%
# 1. 설정값 세밀화
FEES = 0.00015
MAX_LEVERAGE = 2.2  # 기회가 왔을 때 더 공격적으로 (기존 1.8)
STOP_LOSS_THRESHOLD = -0.03 # 최근 3일 누적 수익률이 -3% 이하면 강제 청산 (MDD 방어)

# 2. 데이터 준비
sniper_df = res_df_no.copy()
sniper_df['date'] = pd.to_datetime(sniper_df['date'])

# KOSPI 종가 및 변동성 데이터 결합
df_market = df_raw[['date', 'kospi_close']].copy()
df_market['date'] = pd.to_datetime(df_market['date'])
sniper_df = pd.merge(sniper_df, df_market, on='date', how='left')

# 3. 추가 지표 계산
# 최근 20일 변동성 (변동성이 낮을 때만 레버리지 사용하기 위함)
sniper_df['vol_20d'] = sniper_df['actual'].rolling(20).std() * np.sqrt(252)
# 최근 3일 수익률 (급락장 감지용)
sniper_df['recent_ret_3d'] = sniper_df['actual'].rolling(3).sum()
# 추세 (MA5 & MA20 골든크로스 개념 활용)
sniper_df['ma5'] = sniper_df['kospi_close'].rolling(5).mean()
sniper_df['ma20'] = sniper_df['kospi_close'].rolling(20).mean()

# 4. 스나이퍼 가중치 로직 (핵심)
def get_sniper_weight(row):
    # (1) 강제 손절 및 시장 붕괴 방어 (Tail Risk Guard)
    if row['recent_ret_3d'] < STOP_LOSS_THRESHOLD:
        return 0.0

    # (2) 추세 판별
    is_uptrend = row['kospi_close'] > row['ma5']
    is_strong_uptrend = row['ma5'] > row['ma20']

    # (3) 가중치 결정
    # Case A: 시장이 극도의 공포(KFGI < 25)인데 AI가 상승 예측 -> '스나이퍼' 매수
    if row['kfgi'] < 25 and row['pred'] > 0.001:
        return MAX_LEVERAGE

    # Case B: 상승 추세 구간
    if is_uptrend:
        if row['pred'] > 0:
            # 변동성이 낮을 때만 고레버리지, 높으면 1배만 유지 (샤프지수 관리)
            return MAX_LEVERAGE if row['vol_20d'] < 0.15 else 1.2
        else:
            return 0.8 # AI가 부정적이면 비중 축소

    # Case C: 하락 추세 구간
    else:
        # 하락장이어도 AI 확신도가 매우 높으면 소량 참여, 아니면 관망(0)
        return 0.5 if row['pred'] > 0.003 else 0.0

# 5. 수익률 계산
sniper_df['weight'] = sniper_df.apply(get_sniper_weight, axis=1)

# 수수료 반영 (비중 변경 시 발생)
sniper_df['turnover'] = sniper_df['weight'].diff().abs().fillna(1.0)
sniper_df['transaction_cost'] = sniper_df['turnover'] * FEES
sniper_df['strat_ret_net'] = (sniper_df['weight'] * sniper_df['actual']) - sniper_df['transaction_cost']

# 6. 결과 리포트 함수
def print_comparison(dfs_labels):
    results = []
    for df, label in dfs_labels:
        ann_ret = df['strat_ret_net'].mean() * 252 if 'strat_ret_net' in df.columns else df['actual'].mean() * 252
        ann_vol = df['strat_ret_net'].std() * np.sqrt(252) if 'strat_ret_net' in df.columns else df['actual'].std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)
        cum_ret = np.exp((df['strat_ret_net'] if 'strat_ret_net' in df.columns else df['actual']).cumsum())
        mdd = (cum_ret / cum_ret.cummax() - 1).min()

        results.append({
            'Strategy': label,
            'Ann.Return': f"{ann_ret*100:.2f}%",
            'Sharpe': f"{sharpe:.3f}",
            'MDD': f"{mdd*100:.2f}%"
        })
    return pd.DataFrame(results)

# 결과 출력
print("="*55)
print("🎯 Sniper Hybrid Strategy vs Others")
print("="*55)
comparison_df = print_comparison([
    (res_df, 'Market (Buy & Hold)'),
    (sniper_df, 'Sniper Hybrid (Dynamic)')
])
print(comparison_df)

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(sniper_df['date'], np.exp(sniper_df['actual'].cumsum()), label='Market', color='gray', alpha=0.3)
plt.plot(sniper_df['date'], np.exp(sniper_df['strat_ret_net'].cumsum()), label='Sniper Hybrid', color='forestgreen', lw=2.5)
plt.title("Sharpe Optimization: MDD Defense & Return Boost", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

#%%
# 1. 설정값 세밀화
FEES = 0.00015
MAX_LEVERAGE = 2.2  # 기회가 왔을 때 더 공격적으로 (기존 1.8)
STOP_LOSS_THRESHOLD = -0.03 # 최근 3일 누적 수익률이 -3% 이하면 강제 청산 (MDD 방어)

# 2. 데이터 준비
sniper_d = res_df.copy()
sniper_d['date'] = pd.to_datetime(sniper_d['date'])

# KOSPI 종가 및 변동성 데이터 결합
df_market = df_raw[['date', 'kospi_close']].copy()
df_market['date'] = pd.to_datetime(df_market['date'])
sniper_d = pd.merge(sniper_d, df_market, on='date', how='left')

# 3. 추가 지표 계산
# 최근 20일 변동성 (변동성이 낮을 때만 레버리지 사용하기 위함)
sniper_d['vol_20d'] = sniper_d['actual'].rolling(20).std() * np.sqrt(252)
# 최근 3일 수익률 (급락장 감지용)
sniper_d['recent_ret_3d'] = sniper_d['actual'].rolling(3).sum()
# 추세 (MA5 & MA20 골든크로스 개념 활용)
sniper_d['ma5'] = sniper_d['kospi_close'].rolling(5).mean()
sniper_d['ma20'] = sniper_d['kospi_close'].rolling(20).mean()

# 4. 스나이퍼 가중치 로직 (핵심)
def get_sniper_weight(row):
    # (1) 강제 손절 및 시장 붕괴 방어 (Tail Risk Guard)
    if row['recent_ret_3d'] < STOP_LOSS_THRESHOLD:
        return 0.0

    # (2) 추세 판별
    is_uptrend = row['kospi_close'] > row['ma5']
    is_strong_uptrend = row['ma5'] > row['ma20']

    # (3) 가중치 결정
    # Case A: 시장이 극도의 공포(KFGI < 25)인데 AI가 상승 예측 -> '스나이퍼' 매수
    if row['kfgi'] < 25 and row['pred'] > 0.001:
        return MAX_LEVERAGE

    # Case B: 상승 추세 구간
    if is_uptrend:
        if row['pred'] > 0:
            # 변동성이 낮을 때만 고레버리지, 높으면 1배만 유지 (샤프지수 관리)
            return MAX_LEVERAGE if row['vol_20d'] < 0.15 else 1.2
        else:
            return 0.8 # AI가 부정적이면 비중 축소

    # Case C: 하락 추세 구간
    else:
        # 하락장이어도 AI 확신도가 매우 높으면 소량 참여, 아니면 관망(0)
        return 0.5 if row['pred'] > 0.003 else 0.0

# 5. 수익률 계산
sniper_d['weight'] = sniper_d.apply(get_sniper_weight, axis=1)

# 수수료 반영 (비중 변경 시 발생)
sniper_d['turnover'] = sniper_d['weight'].diff().abs().fillna(1.0)
sniper_d['transaction_cost'] = sniper_d['turnover'] * FEES
sniper_d['strat_ret_net'] = (sniper_d['weight'] * sniper_d['actual']) - sniper_d['transaction_cost']

# 6. 결과 리포트 함수
def print_comparison(dfs_labels):
    results = []
    for df, label in dfs_labels:
        ann_ret = df['strat_ret_net'].mean() * 252 if 'strat_ret_net' in df.columns else df['actual'].mean() * 252
        ann_vol = df['strat_ret_net'].std() * np.sqrt(252) if 'strat_ret_net' in df.columns else df['actual'].std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)
        cum_ret = np.exp((df['strat_ret_net'] if 'strat_ret_net' in df.columns else df['actual']).cumsum())
        mdd = (cum_ret / cum_ret.cummax() - 1).min()

        results.append({
            'Strategy': label,
            'Ann.Return': f"{ann_ret*100:.2f}%",
            'Sharpe': f"{sharpe:.3f}",
            'MDD': f"{mdd*100:.2f}%"
        })
    return pd.DataFrame(results)

# 결과 출력
print("="*55)
print("🎯 Sniper Hybrid Strategy vs Others")
print("="*55)
comparison_d = print_comparison([
    (res_df, 'Market (Buy & Hold)'),
    (sniper_d, 'Sniper Hybrid with sentiment (Dynamic)'),
    (sniper_df, 'Sniper Hybrid without sentiment (Dynamic)')
])
print(comparison_d)

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(sniper_d['date'], np.exp(sniper_d['actual'].cumsum()), label='Market', color='gray', alpha=0.3)
plt.plot(sniper_d['date'], np.exp(sniper_d['strat_ret_net'].cumsum()), label='Sniper Hybrid with sentiment', color='forestgreen', lw=2.5)
plt.plot(sniper_df['date'], np.exp(sniper_df['strat_ret_net'].cumsum()), label='Sniper Hybrid without sentiment', color='red', lw=1.5)
plt.title("Sharpe Optimization: MDD Defense & Return Boost", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()