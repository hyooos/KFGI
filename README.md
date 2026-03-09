# K - Fear & Greed Index

> Korean stock market sentiment index using economic indicators and NAVER comment analysis

## 🍊 BITAmin 16, 17th Winter Project 🍊  
KOSPI 기반 공포·탐욕 지표 및 오실레이터 분석  

NAVER 뉴스 댓글 데이터를 활용한 감성 분석을 통해 **K-Fear & Greed Index (K-FGI)** 를 구축하고  
시계열 예측 및 투자 전략 실험을 수행한 프로젝트입니다.

---

## 📌 Project Overview

- KOSPI 지수와 시장 심리 지표(K-Fear & Greed)를 활용한 시장 분석
- NAVER 뉴스 기사에서 핵심 키워드를 기반으로 댓글 수집 및 감성 분석 수행
- 감성 지표와 거시·기술적 지표를 결합한 시장 심리 지수(K-FGI) 구축
- 시계열 예측 모델을 통해 시장 방향성과 투자 전략 성과 검증

---

## 🎯 Objective

- 기존 **CNN Fear & Greed Index** 구조를 참고하여 한국 시장(KOSPI)에 특화된 공포·탐욕 지표 개발
- 뉴스 댓글 기반 투자 심리를 정량화하여 거시 및 기술적 지표와 통합
- K-FGI 기반 수익률 예측 모델 구축 및 전략 백테스트 수행

---

## 💻 Pipeline

![Pipeline](https://github.com/user-attachments/assets/0d14ddb9-a0ba-40cf-8dc5-1510b4e89f33)

---

## 📊 Result

| Strategy | Annual Return | Sharpe Ratio | MDD |
|--------|---------------|--------------|------|
| Market (Buy & Hold) | 20.14% | 1.057 | -20.67% |
| Original Hybrid (1.8x) | 20.15% | 1.078 | -10.49% |
| **Sniper Hybrid (Dynamic)** | **29.16%** | **1.657** | **-9.50%** |

**Key Insight**

- Sentiment 기반 **K-FGI 지표를 활용한 전략이 가장 높은 수익률과 Sharpe Ratio를 기록**
- Buy & Hold 대비 **리스크(MDD)를 크게 줄이면서 성과 개선**

---

## 📂 Directory Structure

```text
26_winter_proj/
│
├── 1_KFGI_sub_index/        # FGI 기반 7개 sub-index 생성
├── 2_Naver_crawling/        # NAVER 기사 및 댓글 크롤링
├── 3_filtering_final/       # 정치/비주식 댓글 제거
├── 4_sentiment_analysis/    # 금융 특화 감성 분석
├── 5_merge_to_final_csv/    # 감성 + 거시 지표 병합
├── 6_KFGI_weight/           # Ridge 기반 KFGI 가중치 산출
├── 7_modeling_final/        # 시계열 예측 모델
├── 8_dashboard/             # Streamlit 대시보드
│
├── data/
│   ├── KFG/
│   └── NAVER/
│       ├── article/
│       ├── comments/
│       ├── final_filtered/
│       └── sentiment_final/
│
├── documents/
├── README.md
└── .gitignore
```

## 🧪 Environment
### 🐍 언어
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  
### 📚 주요 패키지
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-2CA5E0?style=for-the-badge&logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4B8BBE?style=for-the-badge&logo=python&logoColor=white)


## 🧠 Methodology

**1️⃣ Sub-Index Construction**
* KOSPI 기반 7개 시장 심리 지표 수집
* 0–100 정규화
* 시계열 정렬 및 결측 처리

**2️⃣ Sentiment Extraction**
* NAVER 뉴스 기사 크롤링
* 댓글 필터링 (정치/비주식/독성 제거)
* 금융 특화 BERT 기반 감성 확률 산출
* 일별 감성지표 생성

**3️⃣ Index Aggregation**
* Ridge 회귀 기반 가중치 추정
* PCA / Factor Analysis 비교
* K-FGI 지수 산출

**4️⃣ Forecasting & Strategy**
* Multi-horizon 예측 실험
* Directional Accuracy 분석
* 전략 수익률 및 샤프지수 비교


**5️⃣ Dashboard Visualization**
* 실행 : 8_dashboard/app.py
* 입력 : KFGI_final.csv, 예측 결과 파일
* 출력 : Streamlit 기반 시각화 대시보드, KFGI 지수 추이 및 투자 전략 결과 확인
