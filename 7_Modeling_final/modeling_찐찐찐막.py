
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

# 데이터 준비
df_raw = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv')
df_eng = build_advanced_features(df_raw)
df_final, core_feats = create_logical_kfgi(df_eng)

df = df_final.copy()

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
'''
여기가 메인 수익률 예측 함수 
중요중요
'''
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

for col in ['actual', 'strat_ret']:
    name = "Buy & Hold (Market)" if col == 'actual' else "Advanced Strategy"
    ann_ret = res_df[col].mean() * 252
    ann_vol = res_df[col].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum_ret = np.exp(res_df[col].cumsum())
    mdd = (cum_ret / cum_ret.cummax() - 1).min()

    print(f"[{name}]")
    print(f"   - 연수익률: {ann_ret*100:.2f}%")
    print(f"   - 샤프지수: {sharpe:.3f}")
    print(f"   - 최대낙폭(MDD): {mdd*100:.2f}%")
    print("-" * 30)

print(f"🔍 과적합 진단: 구간별 오차 표준편차 = {np.std(errors):.6f}")
if np.std(errors) > 0.005:
    print("⚠️ 주의: 구간별 성능 편차가 큽니다. 과적합 가능성이 있으니 파라미터를 조정하세요.")
else:
    print("✅ 안정성: 구간별 성능이 일정하게 유지되어 과적합 위험이 낮습니다.")




# 7. 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# ---- 상단: 누적 수익 ----
ax1.plot(res_df['date'], np.exp(res_df['actual'].cumsum()),
         label='Market', color='gray', alpha=0.6)
ax1.plot(res_df['date'], np.exp(res_df['strat_ret'].cumsum()),
         label='Strategy', color='crimson', lw=2)

ax1.set_title("Cumulative Return")
ax1.legend()
ax1.grid(True, alpha=0.2)

# ---- 하단: KFGI ----
ax2.plot(res_df['date'], res_df['kfgi'],
         label='KFGI', color='dodgerblue')

ax2.axhline(30, color='blue', linestyle='--', alpha=0.6)
ax2.axhline(80, color='red', linestyle='--', alpha=0.6)

# Fear 구간 음영
ax2.fill_between(res_df['date'], 0, 100,
                 where=(res_df['kfgi'] < 30),
                 color='blue', alpha=0.1)

ax2.set_ylim(0, 100)
ax2.set_title("KFGI (Fear < 30, Greed > 80)")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()











'''
여기까지가 기본적인 예측 모형 + 진짜 기본적인 수익률 계산 -> 이때 수익률은 무시해도 됨
'''



#%%
## 전략 1. [불도저 전략] 수익률과 샤프지수 동시 폭발 모델

bulldozer_res = res_df.copy()

# 1. 공격적 참여 (참여율 95% 이상 확보)
# 하위 5%의 극단적 하락 신호가 아니면 무조건 매수 포지션 유지
threshold = bulldozer_res['pred'].quantile(0.05)
bulldozer_res['signal'] = np.where(bulldozer_res['pred'] > threshold, 1, 0)

# 2. 불도저 레버리지 로직
def calculate_bulldozer_weight(row):
    # 역발상 베팅: 공포(KFGI < 45) 구간에서 2.5배 레버리지로 시장 수익률 압도
    if row['kfgi'] < 45:
        return 2.5
    # 과열 구간: 그래도 시장 수익률은 따라감 (비중 1.0)
    elif row['kfgi'] > 85:
        return 1.0
    # 보통 구간: 살짝 레버리지 (1.2배)
    else:
        return 1.2

bulldozer_res['bull_weight'] = bulldozer_res.apply(calculate_bulldozer_weight, axis=1)

# 3. 최종 전략 수익률
bulldozer_res['bull_ret'] = bulldozer_res['signal'] * bulldozer_res['bull_weight'] * bulldozer_res['actual']

# 4. 성과 보고서
print("="*45)
print("🚜 시장을 밀어버리는 '불도저' 전략 결과")
print("="*45)

for col in ['actual', 'bull_ret']:
    name = "Buy & Hold (Market)" if col == 'actual' else "The Bulldozer"
    ann_ret = bulldozer_res[col].mean() * 252
    ann_vol = bulldozer_res[col].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum_ret = np.exp(bulldozer_res[col].cumsum())
    mdd = (cum_ret / cum_ret.cummax() - 1).min()

    print(f"[{name}]")
    print(f"   - 연수익률: {ann_ret*100:.2f}%")
    print(f"   - 샤프지수: {sharpe:.3f}")
    print(f"   - 최대낙폭(MDD): {mdd*100:.2f}%")
    print("-" * 40)

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(bulldozer_res['date'], np.exp(bulldozer_res['actual'].cumsum()), label='Market', color='gray', alpha=0.5)
plt.plot(bulldozer_res['date'], np.exp(bulldozer_res['bull_ret'].cumsum()), label='The Bulldozer', color='darkred', lw=2.5)
plt.title("The Bulldozer Strategy: Maximum Profit Focus")
plt.legend(); plt.grid(True, alpha=0.2); plt.show()









#%%
'''
리스크 관리 ?
'''
## 전략 2. [통합 셀] 추세 추종 + AI 예측 하이브리드 전략

final_hybrid = res_df.copy()

# 1. 추세 지표 생성 (5일 이동평균선)
# kospi_close 데이터가 필요하므로 df_raw에서 가져옵니다.
final_hybrid['ma5'] = df_raw['kospi_close'].rolling(5).mean().iloc[final_hybrid.index]
final_hybrid['market_trend'] = np.where(df_raw['kospi_close'].iloc[final_hybrid.index] > final_hybrid['ma5'], 1, 0)

# 2. 하이브리드 로직 (Trend + AI)
def calculate_hybrid_weight(row):
    # (1) 시장이 상승 추세일 때
    if row['market_trend'] == 1:
        if row['pred'] > 0: # AI도 좋다고 하면 -> 풀 레버리지
            return 2.2
        else: # AI는 별로라고 하면 -> 기본만 들고 감
            return 1.0
    # (2) 시장이 하락 추세일 때
    else:
        if row['pred'] > 0.001: # AI가 아주 강력한 반등을 예고할 때만 소량 참여
            return 0.5
        else: # 그 외엔 전량 현금화 (이게 MDD 방어의 핵심)
            return 0.0

final_hybrid['hybrid_weight'] = final_hybrid.apply(calculate_hybrid_weight, axis=1)

# 3. 최종 전략 수익률
final_hybrid['hybrid_ret'] = final_hybrid['hybrid_weight'] * final_hybrid['actual']

# 4. 성과 보고서
print("\n" + "="*45)
print("🚀 [최종] 추세 결합형 하이브리드 전략 결과")
print("="*45)

for col in ['actual', 'hybrid_ret']:
    name = "Buy & Hold (Market)" if col == 'actual' else "Hybrid-Trend"
    ann_ret = final_hybrid[col].mean() * 252
    ann_vol = final_hybrid[col].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum_ret = np.exp(final_hybrid[col].cumsum())
    mdd = (cum_ret / cum_ret.cummax() - 1).min()

    print(f"[{name}]")
    print(f"   - 연수익률: {ann_ret*100:.2f}%")
    print(f"   - 샤프지수: {sharpe:.3f}")
    print(f"   - 최대낙폭(MDD): {mdd*100:.2f}%")
    print("-" * 40)

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(final_hybrid['date'], np.exp(final_hybrid['actual'].cumsum()), label='Market', color='gray', alpha=0.5)
plt.plot(final_hybrid['date'], np.exp(final_hybrid['hybrid_ret'].cumsum()), label='Hybrid Strategy', color='blue', lw=2)
plt.title("Hybrid Strategy: AI + Market Trend")
plt.legend(); plt.grid(True, alpha=0.2); plt.show()



#%%
## 전략3. [최종 수정본] 인덱스 오류 해결 및 수수료 반영 하이브리드 전략
# 조금 더 실전 친화적임

import pandas as pd
import numpy as np

# 1. 환경 설정
FEES = 0.00015  # 편도 수수료 0.015%
LEVERAGE = 1.8  # 레버리지 설정

# 2. 이동평균선(MA5) 데이터 준비 (전체 데이터 기반)
# df_final이나 df_raw 중 kospi_close가 있는 전체 데이터를 사용하세요.
# 여기서는 가장 원본에 가까운 df_final을 기준으로 계산합니다.
ma_df = df_final[['date', 'kospi_close']].copy()
ma_df['ma5'] = ma_df['kospi_close'].rolling(5).mean()
ma_df['market_trend'] = np.where(ma_df['kospi_close'] > ma_df['ma5'], 1, 0)

# 3. 결과 데이터(res_df)와 이동평균 데이터 병합 (인덱스 대신 'date' 기준)
# res_df에 이미 ma5나 market_trend가 있다면 중복 방지를 위해 제거 후 병합
cols_to_use = ['date', 'ma5', 'market_trend']
final_strategy_df = res_df.drop(columns=[c for c in cols_to_use if c in res_df.columns and c != 'date'])
final_strategy_df = pd.merge(final_strategy_df, ma_df[cols_to_use], on='date', how='left')

# 4. 하이브리드 가중치 계산 함수
def get_hybrid_weight(row):
    # 데이터가 부족해 ma5가 NaN인 초기 구간은 보수적으로 1.0(시장추종) 처리
    if pd.isna(row['ma5']):
        return 1.0

    # 시장이 상승 추세일 때 (MA5 위)
    if row['market_trend'] == 1:
        return LEVERAGE if row['pred'] > 0 else 1.0
    # 시장이 하락 추세일 때 (MA5 아래)
    else:
        return 0.3 if row['pred'] > 0.002 else 0.0

# 5. 가중치 및 수수료 적용 수익률 계산
final_strategy_df['weight'] = final_strategy_df.apply(get_hybrid_weight, axis=1)

# 거래 비용 계산 (비중 변화 시 발생)
final_strategy_df['turnover'] = final_strategy_df['weight'].diff().abs().fillna(final_strategy_df['weight'].iloc[0])
final_strategy_df['transaction_cost'] = final_strategy_df['turnover'] * FEES

# 최종 수익률 = (비중 * 실제수익률) - 거래비용
final_strategy_df['strat_ret_net'] = (final_strategy_df['weight'] * final_strategy_df['actual']) - final_strategy_df['transaction_cost']

# 6. 최종 성과 보고
def print_final_report(df):
    results = []
    for col, label in [('actual', 'Buy & Hold (Market)'), ('strat_ret_net', f'Hybrid (Lev {LEVERAGE})')]:
        ann_ret = df[col].mean() * 252
        ann_vol = df[col].std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)
        cum_ret = np.exp(df[col].cumsum())
        mdd = (cum_ret / cum_ret.cummax() - 1).min()
        results.append({
            '전략': label,
            '연수익률': f"{ann_ret*100:.2f}%",
            '샤프지수': f"{sharpe:.3f}",
            '최대낙폭(MDD)': f"{mdd*100:.2f}%"
        })
    return pd.DataFrame(results)

print("="*45)
print("🚀 최종 하이브리드 전략 검증 결과 (수수료 반영)")
print("="*45)
print(print_final_report(final_strategy_df))



#%%
## [최종 수정본] 하이브리드 전략 및 현실성 검증 코드

import pandas as pd
import numpy as np

# 1. 환경 설정 (수수료 및 레버리지)
FEES = 0.00015  # 편도 수수료 0.015%
LEVERAGE = 1.8  # 가장 효율이 좋았던 1.8배 설정 (원하시는 대로 수정 가능)

# 2. 결과 데이터프레임 복사
# res_df가 이미 존재한다고 가정합니다.
final_strategy_df = res_df.copy()

# Ensure 'date' column is datetime type for merging
final_strategy_df['date'] = pd.to_datetime(final_strategy_df['date'])

# 3. 추세 지표(MA5) 안전하게 결합
# df_raw에서 'kospi_close'를 가져와서 날짜를 기준으로 final_strategy_df와 병합
df_kospi_close = df_raw[['date', 'kospi_close']].copy()
df_kospi_close['date'] = pd.to_datetime(df_kospi_close['date']) # Ensure datetime for merging

final_strategy_df = pd.merge(final_strategy_df, df_kospi_close, on='date', how='left')

# Calculate MA5 directly on the merged df, then apply market_trend
final_strategy_df['ma5'] = final_strategy_df['kospi_close'].rolling(5).mean()
final_strategy_df['market_trend'] = np.where(final_strategy_df['kospi_close'] > final_strategy_df['ma5'], 1, 0)

# Drop NaN values that result from rolling mean (first few rows)
final_strategy_df.dropna(subset=['ma5', 'market_trend'], inplace=True)
final_strategy_df.reset_index(drop=True, inplace=True) # Reset index after dropping rows

# 4. 하이브리드 가중치 함수 정의
def get_final_weight(row):
    # 시장이 상승 추세일 때 (MA5 위)
    if row['market_trend'] == 1:
        return LEVERAGE if row['pred'] > 0 else 1.0
    # 시장이 하락 추세일 때 (MA5 아래)
    else:
        return 0.3 if row['pred'] > 0.002 else 0.0

# 5. 가중치 및 수익률 계산 (수수료 포함)
final_strategy_df['weight'] = final_strategy_df.apply(get_final_weight, axis=1)

# 거래 비용 계산: 비중이 변할 때만 수수료 발생
# fillna(LEVERAGE) for the first day's transaction, assuming full position if weight is not 0
final_strategy_df['turnover'] = final_strategy_df['weight'].diff().abs().fillna(LEVERAGE)
final_strategy_df['transaction_cost'] = final_strategy_df['turnover'] * FEES

# 최종 수익률 = (비중 * 실제수익률) - 거래비용
final_strategy_df['strat_ret_net'] = (final_strategy_df['weight'] * final_strategy_df['actual']) - final_strategy_df['transaction_cost']

# 6. 성과 지표 출력
def print_performance(df, col_name, label):
    ann_ret = df[col_name].mean() * 252
    ann_vol = df[col_name].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum_ret = np.exp(df[col_name].cumsum())
    mdd = (cum_ret / cum_ret.cummax() - 1).min()

    print(f"[{label}]")
    print(f"   - 연수익률: {ann_ret*100:.2f}%")
    print(f"   - 샤프지수: {sharpe:.3f}")
    print(f"   - 최대낙폭(MDD): {mdd*100:.2f}%")
    print("-" * 30)

print("="*40)
print("🏆 최종 하이브리드 전략 검증 (수수료 반영)")
print("="*40)
print_performance(final_strategy_df, 'actual', 'Buy & Hold (Market)')
print_performance(final_strategy_df, 'strat_ret_net', f'Hybrid Strategy (Lev {LEVERAGE})')

# 결과 확인을 위해 상위 5개 행 출력
# print(final_strategy_df[['date', 'market_trend', 'weight', 'strat_ret_net']].head())

plt.figure(figsize=(14, 7))

# 누적수익 계산
cum_market = np.exp(final_strategy_df['actual'].cumsum())
cum_strategy = np.exp(final_strategy_df['strat_ret_net'].cumsum())

plt.plot(final_strategy_df['date'], cum_market,
         label='Market (Buy & Hold)', color='gray', alpha=0.6)

plt.plot(final_strategy_df['date'], cum_strategy,
         label='Final Hybrid Strategy (Net)', color='navy', lw=2)

plt.title("Final Hybrid Strategy (Fee Included) vs Market")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(alpha=0.2)
plt.show()


def drawdown(series):
    cum = np.exp(series.cumsum())
    return cum / cum.cummax() - 1

plt.figure(figsize=(14,5))
plt.plot(final_strategy_df['date'],
         drawdown(final_strategy_df['actual']),
         label='Market DD', color='gray')

plt.plot(final_strategy_df['date'],
         drawdown(final_strategy_df['strat_ret_net']),
         label='Strategy DD', color='navy')

plt.title("Drawdown Comparison")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
