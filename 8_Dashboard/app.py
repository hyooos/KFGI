#%%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. 페이지 기본 설정 및 커스텀 CSS 적용
# ----------------------------------------------------------------------
st.set_page_config(page_title="K-FGI Strategy", layout="wide", page_icon="📈")

st.markdown("""
    <style>
        .main .block-container h1 { font-size: 3rem; font-weight: 800; color: #00adb5; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin-bottom: 0px; }
        .main .block-container p { font-size: 1.2rem; font-weight: 500; color: #aaaaaa; }
        h3 { font-weight: 700 !important; color: #eeeeee !important; border-bottom: 2px solid #00adb5; padding-bottom: 10px; margin-top: 30px !important; }
        [data-testid="stMetricValue"] { font-size: 2rem !important; color: #00adb5 !important; font-weight: 700 !important; }
        [data-testid="stMetricDelta"] svg { fill: #eeeeee !important; }
        hr { margin: 2em 0px; border-top: 1px solid #444444 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡Panic Buy, Party Sell")
st.markdown(": 감정을 지배하는 투자 전략")
st.divider()

# ----------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# ----------------------------------------------------------------------
@st.cache_data
def walk_forward_prediction(df, features, target, train_ratio=0.6):
    df = df.copy().reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    preds = []
    dates = []
    actuals = []

    print("🚀 Walk-forward 학습 진행 중...")

    for i in range(split_idx, len(df)):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]

        X_train = train_df[features].fillna(0)
        y_train = train_df[target]

        X_test = test_df[features].fillna(0)

        # Ridge
        model_r = Ridge(alpha=1.0)
        model_r.fit(X_train, y_train)

        # LGBM (과적합 방지용 약한 세팅)
        model_l = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            verbosity=-1
        )
        model_l.fit(X_train, y_train)

        pred = 0.2 * model_r.predict(X_test)[0] + \
               0.8 * model_l.predict(X_test)[0]

        preds.append(pred)
        dates.append(test_df['date'].values[0])
        actuals.append(test_df[target].values[0])

    result = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'pred': preds,
        'actual': actuals
    })

    return result

def load_and_prepare_data():
    # 개인 로컬에 맞게 수정
    df = pd.read_csv('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True).ffill().bfill()
    
    df['sent_energy'] = df['sent_strength_w'] * df['sent_norm_w']
    df['neg_z_inv'] = -df['neg_z']
    for i in range(1, 8): df[f'sub_index{i}_lag1'] = df[f'sub_index{i}'].shift(1)
    
    df['target'] = df['log_return_t+1']
    df = df.dropna().reset_index(drop=True)
    
    core_feats = [f'sub_index{i}' for i in range(1, 8)] + ['sent_norm_w', 'sent_energy', 'neg_z_inv']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[core_feats])
    
    ridge = RidgeCV().fit(X_scaled, df['target'])
    weights = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))
    
    raw_fgi = X_scaled @ weights
    p1, p99 = np.percentile(raw_fgi, [1, 99])
    df['KFGI'] = 100 * (np.clip(raw_fgi, p1, p99) - p1) / (p99 - p1)
    
    df['ma5'] = df['kospi_close'].rolling(5).mean()
    df['market_trend'] = np.where(df['kospi_close'] > df['ma5'], 1, 0)
    df = df.dropna().reset_index(drop=True)

    exclude_cols = ['date', 'kospi_close', 'log_return', 'log_return_t+1', 'actual_1d', 'target_reg', 'target_3d', 'target_5d', 'KFGI', 'target','target_cls']
    
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    res_df = walk_forward_prediction(df, features, 'target')

    # OOS 예측만 병합
    df = df.merge(res_df[['date', 'pred']], on='date', how='left')

    # 학습 초기 구간 제거 (완전 OOS만 사용)
    df = df.dropna(subset=['pred']).reset_index(drop=True)

    # -------------------------------
    # 🔥 미래 정보 없는 threshold 계산
    # (expanding quantile)
    # -------------------------------
    df['threshold'] = (
        df['pred']
        .expanding()
        .apply(lambda x: np.quantile(x, 0.05), raw=True)
    )

    df['bull_signal'] = np.where(df['pred'] > df['threshold'], 1, 0)

    df['bull_weight'] = df['KFGI'].apply(
        lambda x: 2.5 if x < 25 else (1.0 if x > 75 else 1.2)
    )

    df['bull_ret'] = df['bull_signal'] * df['bull_weight'] * df['target']

    # -------------------------------
    # 🎯 Sniper Hybrid 전략 (Dynamic)
    # -------------------------------

    FEES = 0.00015
    MAX_LEVERAGE = 2.2
    STOP_LOSS_THRESHOLD = -0.03

    # 추가 지표 계산
    df['vol_20d'] = df['target'].rolling(20).std() * np.sqrt(252)
    df['recent_ret_3d'] = df['target'].rolling(3).sum()
    df['ma20'] = df['kospi_close'].rolling(20).mean()

    def get_sniper_weight(row):

        # 1️⃣ Tail Risk Guard
        if row['recent_ret_3d'] < STOP_LOSS_THRESHOLD:
            return 0.0

        # 2️⃣ 추세 판단
        is_uptrend = row['kospi_close'] > row['ma5']
        is_strong_uptrend = row['ma5'] > row['ma20']

        # 3️⃣ Extreme Fear + AI 상승
        if row['KFGI'] < 25 and row['pred'] > 0.001:
            return MAX_LEVERAGE

        # 4️⃣ 상승 추세
        if is_uptrend:
            if row['pred'] > 0:
                return MAX_LEVERAGE if row['vol_20d'] < 0.15 else 1.2
            else:
                return 0.8

        # 5️⃣ 하락 추세
        else:
            return 0.5 if row['pred'] > 0.003 else 0.0


    df['hyb_weight'] = df.apply(get_sniper_weight, axis=1)

    df['hyb_turnover'] = df['hyb_weight'].diff().abs().fillna(1.0)

    df['hyb_ret'] = (
        df['hyb_weight'] * df['target']
        - df['hyb_turnover'] * FEES
    )

    df['bnh_ret'] = df['target']

    return df

df = load_and_prepare_data()

# ----------------------------------------------------------------------
# 3. 사이드바 구성 (버튼 제어 및 금액 포맷 적용)
# ----------------------------------------------------------------------
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

st.sidebar.title("K-Fear & Greed Index")
st.sidebar.markdown("---")

with st.sidebar.form(key='setup_form'):
    strategy_choice = st.radio("1️⃣ 투자 전략 선택", ('스나이퍼 (MDD 방어형)', '불도저 (공격/위험감수형)'))
    position_choice = st.radio("2️⃣ 현재 포지션", ('매수 포지션', '매도 포지션'))
    
    capital_options = {
        "100만 원 (1,000,000)": 1000000,
        "500만 원 (5,000,000)": 5000000,
        "1,000만 원 (10,000,000)": 10000000,
        "5,000만 원 (50,000,000)": 50000000,
        "1억 원 (100,000,000)": 100000000,
        "5억 원 (500,000,000)": 500000000
    }
    selected_capital_str = st.selectbox("3️⃣ 초기 투자 금액", options=list(capital_options.keys()), index=2)
    initial_capital = capital_options[selected_capital_str]
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date, end_date = st.slider("4️⃣ 백테스팅 기간 설정", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    
    st.markdown("---")
    submit_button = st.form_submit_button(label="분석 시작", use_container_width=True)

if submit_button:
    st.session_state.analysis_started = True

# ----------------------------------------------------------------------
# 4. 메인 화면 제어 (안내화면 vs 분석화면)
# ----------------------------------------------------------------------
if not st.session_state.analysis_started:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("👈 **왼쪽 메뉴에서 투자 성향과 포지션을 설정한 후 [분석 시작] 버튼을 눌러주세요.**", icon="ℹ️")
    st.markdown("""
        <div style='text-align: center; color: #aaaaaa; margin-top: 20px;'>
            <h3>시장의 심리를 읽고, 데이터로 증명합니다.</h3>
            <p>K-FGI(한국형 공포탐욕지수)를 활용한 당신만의 맞춤형 퀀트 리포트를 생성해보세요.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # --- 여기서부터 들여쓰기(Indentation) 되어있어야 안전하게 실행됩니다 ---
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    filtered_df = df.loc[mask].copy()
    strat_col = 'hyb_ret' if '스나이퍼' in strategy_choice else 'bull_ret'
    filtered_df['cum_strat'] = (1 + filtered_df[strat_col]).cumprod()
    filtered_df['cum_bnh'] = (1 + filtered_df['bnh_ret']).cumprod()
    filtered_df['strat_asset'] = initial_capital * filtered_df['cum_strat']
    filtered_df['bnh_asset'] = initial_capital * filtered_df['cum_bnh']

    # --- 섹션 A ---
    st.subheader(f"💡 맞춤형 시장 진단 & 액션 플랜 ({position_choice.split(' ')[1]})")

    latest_data = filtered_df.iloc[-1]
    current_kfgi = latest_data['KFGI']
    current_trend = "상승 추세 📈" if latest_data['market_trend'] == 1 else "하락 추세 📉"
    next_weight = latest_data['hyb_weight'] if '스나이퍼' in strategy_choice else latest_data['bull_weight']

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = current_kfgi,
            title = {'text': "<b>K-FGI</b> (공포탐욕지수)", 'font': {'color': '#eeeeee', 'size': 20}},
            number = {'font': {'color': '#eeeeee', 'size': 40}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': '#eeeeee'},
                'bar': {'color': "#eeeeee", 'thickness': 0.15}, 
                'bgcolor': "#222831", 'bordercolor': "#444444",
                'steps': [
                    {'range': [0, 25], 'color': "#00adb5"}, {'range': [25, 45], 'color': "#00565b"},
                    {'range': [45, 55], 'color': "#444444"}, {'range': [55, 75], 'color': "#8a3011"},
                    {'range': [75, 100], 'color': "#ff5722"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.8, 'value': current_kfgi}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#eeeeee'})
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

    with col2:
        st.markdown(f"###  포지션 맞춤 가이드")
        if current_kfgi < 25:
            state_text, status_color = "🥶 극단적 공포 (Extreme Fear)", "#00adb5"
            buy_guide, sell_guide = "적극 매수 기회! 시장의 패닉을 역이용하세요.", "투매(패닉셀) 절대 금지! 바닥에서 파는 격입니다."
        elif current_kfgi < 45:
            state_text, status_color = "😨 공포 (Fear)", "#007a80"
            buy_guide, sell_guide = "저가 분할 매수 접근. 좋은 가격 구간입니다.", "손절 자제. 기술적 반등을 인내하세요."
        elif current_kfgi < 55:
            state_text, status_color = "😐 중립 (Neutral)", "#aaaaaa"
            if latest_data['market_trend'] == 0:
                buy_guide, sell_guide = "관망 (하락 추세 지속). 반등 시그널 대기.", "반등 시 기계적 비중 축소 및 리스크 관리."
            else:
                buy_guide, sell_guide = "단기 추세에 편승한 유연한 진입 가능.", "상승 추세 이탈 전까지 지속 홀딩."
        elif current_kfgi < 75:
            state_text, status_color = "😏 탐욕 (Greed)", "#c74318"
            buy_guide, sell_guide = "신규 진입 비중 축소. 상승 여력 제한적.", "분할 익절을 시작하여 현금 비중 확대."
        else: 
            state_text, status_color = "🔥 극단적 탐욕 (Extreme Greed)", "#ff5722"
            buy_guide, sell_guide = "추격 매수(FOMO) 절대 금지! 시장 과열 징후.", "적극적 익절! 이익 실현 및 현금화."

        st.info(f"**현재 시장 상태:** {state_text} / **추세:** {current_trend}")
        
        if '매수' in position_choice:
            st.markdown(f"""<div style="background-color: #2b303a; padding: 15px; border-left: 5px solid {status_color}; border-radius: 5px; margin-top: 10px;">
                <h4 style="margin-top: 0px; color: {status_color};"> 매수 타이밍 진단</h4><p style="font-size: 1.1em; color: #eeeeee; margin-bottom: 0px;">👉 <b>{buy_guide}</b></p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #2b303a; padding: 15px; border-left: 5px solid {status_color}; border-radius: 5px; margin-top: 10px;">
                <h4 style="margin-top: 0px; color: {status_color};"> 매도/홀딩 진단</h4><p style="font-size: 1.1em; color: #eeeeee; margin-bottom: 0px;">👉 <b>{sell_guide}</b></p></div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"###  시스템 권장 비중")
        st.metric(
            label=f"{strategy_choice.split(' ')[0]} 전략 목표 비중", value=f"{next_weight}x", 
            delta=f"전일 대비 {latest_data[strat_col.replace('_ret','_weight')] - filtered_df.iloc[-2][strat_col.replace('_ret','_weight')]:.1f}x 변동"
        )
        
        # 선택한 전략에 따른 설명 텍스트
        if '하이브리드' in strategy_choice:
            strat_desc = "📉 <b>[방어 우선]</b> 심리지표보다 코스피 단기 추세와 모델 예측을 우선하여 비중을 조절합니다."
        else:
            strat_desc = "🔥 <b>[역발상 우선]</b> K-FGI의 극단적 심리 상태를 역이용하여 공격적으로 비중을 산출합니다."

        if '현금' in position_choice:
            if next_weight == 0: action_text = "매수 보류 및 관망 (리스크 회피) 💸"
            elif next_weight < 1: action_text = f"자본금의 {int(next_weight*100)}%만 제한적 매수 🛡️"
            elif next_weight == 1: action_text = "자본금 100% 정상 매수 📈"
            else: action_text = f"자본금의 {int(next_weight*100)}% 강력 매수 (레버리지) 🔥"
        else: # 주식 보유자인 경우
            if next_weight == 0: 
                action_text = "시스템 강제 청산 (전량 매도) 💸"
                # 논리적 충돌(공포인데 매도 지시)이 발생할 경우 부연 설명 추가
                if current_kfgi < 45:
                    action_text += "<br><span style='font-size: 0.8em; color: #ff5722;'>⚠️ K-FGI는 공포 구간이나, 모델이 추가 하락(추세 붕괴)을 강력히 예측하여 기계적 손절을 지시했습니다.</span>"
            elif next_weight < 1: 
                action_text = f"주식 비중 {int(next_weight*100)}%로 축소 (부분 익절/손절) 🛡️"
            elif next_weight == 1: 
                action_text = "현재 주식 포지션 유지 (홀딩) 📈"
            else: 
                action_text = f"매도 보류 및 레버리지 확대 🔥"
            
        st.markdown(f"<div style='text-align: center; font-size: 1.1em; font-weight: bold; padding: 10px; background-color: #393e46; border-radius: 5px;'>{action_text}</div>", unsafe_allow_html=True)
        st.caption(f"※ 1x=100% 투자, >1x=레버리지, 0x=현금 관망<br>{strat_desc}", unsafe_allow_html=True)

    st.divider()

    # --- 섹션 B ---
    st.subheader(f"📊 {strategy_choice.split(' ')[0]} 전략 백테스팅 성과")

    total_return = (filtered_df['cum_strat'].iloc[-1] - 1) * 100
    bnh_return = (filtered_df['cum_bnh'].iloc[-1] - 1) * 100

    def get_mdd(series):
        roll_max = series.cummax()
        drawdown = series / roll_max - 1.0
        return drawdown.min() * 100

    strat_mdd = get_mdd(filtered_df['cum_strat'])
    bnh_mdd = get_mdd(filtered_df['cum_bnh'])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("최종 자산 평가액", f"{int(filtered_df['strat_asset'].iloc[-1]):,} 원", f"{total_return:+.1f}% 수익")
    kpi2.metric("비교지수(코스피) 수익률", f"{bnh_return:+.1f}%", delta_color="off")
    kpi3.metric("최대 낙폭(MDD) 방어", f"{strat_mdd:.1f}%", "리스크 관리 핵심", delta_color="inverse")
    kpi4.metric("코스피 MDD", f"{bnh_mdd:.1f}%", delta_color="inverse")

    common_layout = dict(
        hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eeeeee'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#444444', zeroline=False, showline=True, linewidth=1, linecolor='#666666'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#444444', zeroline=False, showline=True, linewidth=1, linecolor='#666666'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#444444', borderwidth=1)
    )

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['strat_asset'], mode='lines', name=f"<b>{strategy_choice.split(' ')[0]} 전략</b>", line=dict(color='#00adb5', width=3)))
    fig_equity.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['bnh_asset'], mode='lines', name='KOSPI 단순보유', line=dict(color='#777777', width=1.5, dash='dot')))
    fig_equity.update_layout(title="<b>누적 자산 성장 추이</b> (Equity Curve)", yaxis_title="자산 평가액 (원)", height=450, **common_layout)
    st.plotly_chart(fig_equity, use_container_width=True, key="equity")

    fig_kfgi = go.Figure()
    fig_kfgi.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['kospi_close'], name="KOSPI 지수", line=dict(color='#aaaaaa', width=1.5)))
    fig_kfgi.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['KFGI'], name="<b>K-FGI 지수</b>", yaxis="y2", line=dict(color='#ff5722', width=2), opacity=0.8)) 

    fig_kfgi.add_hrect(y0=0, y1=25, yref="y2", fillcolor="#00adb5", opacity=0.15, line_width=0, layer="below", annotation_text="극단적 공포 (적극 매수)", annotation_font_color='#00adb5')
    fig_kfgi.add_hrect(y0=75, y1=100, yref="y2", fillcolor="#ff5722", opacity=0.15, line_width=0, layer="below", annotation_text="극단적 탐욕 (적극 매도)", annotation_font_color='#ff5722', annotation_position="top left")

    kfgi_layout = common_layout.copy()
    kfgi_layout.update(dict(
        title="<b>코스피 흐름과 K-FGI 심리 국면</b>",
        yaxis=dict(title="KOSPI 지수", **common_layout['yaxis']),
        yaxis2=dict(title="K-FGI (0~100)", overlaying="y", side="right", range=[0, 100], showgrid=False, linecolor='#666666', tickfont=dict(color='#ff5722')),
        height=450
    ))
    fig_kfgi.update_layout(**kfgi_layout)
    st.plotly_chart(fig_kfgi, use_container_width=True, key="kfgi")