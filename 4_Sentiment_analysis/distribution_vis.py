#%%
import pandas as pd
import os

# 개인 환경에 맞게 수정 필요
base_dir = '/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/sentiment_scores'

years = [2023, 2024, 2025]

dfs = {}

'''
기본 통계량
'''
for y in years:
    path = os.path.join(base_dir, f'comments_sentiment_{y}.csv')
    df = pd.read_csv(path)
    dfs[y] = df
    
    print(f"\n===== {y} =====")
    print(df["sentiment_score"].describe())
    print("부정 비율:", (df["sentiment_score"] < 0).mean())
    print("긍정 비율:", (df["sentiment_score"] > 0).mean())

#%%
'''
감성 점수 분포 시각화
'''
import matplotlib.pyplot as plt

for y in years:
    plt.figure(figsize=(6,4))
    plt.hist(dfs[y]["sentiment_score"], bins=50)
    plt.axvline(0, linestyle="--")
    plt.title(f"Sentiment Distribution - {y}")
    plt.xlabel("sentiment_score")
    plt.ylabel("count")
    plt.show()

#%%
'''
감성 점수 분포 시각화 (KDE)
'''
import seaborn as sns  

for y in years:
    plt.figure(figsize=(6,4))
    sns.histplot(dfs[y]["sentiment_score"], bins=50, kde=True)
    plt.axvline(0, linestyle="--")
    plt.title(f"Sentiment Distribution with KDE - {y}")
    plt.xlabel("sentiment_score")
    plt.ylabel("count")
    plt.show()

#%%
'''
연도별 감성 점수 분포 비교
'''

import seaborn as sns

plt.figure(figsize=(8,6))

for y in years:
    sns.kdeplot(dfs[y]["sentiment_score"], label=str(y), fill=False)

plt.axvline(0, linestyle="--")
plt.title("Sentiment Distribution Comparison")
plt.legend()
plt.show()

#%%
'''
박스플롯
'''
combined = []

for y in years:
    temp = dfs[y][["sentiment_score"]].copy()
    temp["year"] = y
    combined.append(temp)

combined = pd.concat(combined)

plt.figure(figsize=(8,5))
sns.boxplot(data=combined, x="year", y="sentiment_score")
plt.axhline(0, linestyle="--")
plt.title("Sentiment Distribution by Year")
plt.show()

#%%
# 상위 20개 댓글
df_sorted=[1,1,1]
for i, y in enumerate(years):
    df_sorted[i] = dfs[y].sort_values("sentiment_score")

for i in range(len(years)):
    print(f"\n===== {years[i]} 상위 20개 =====")
    print(df_sorted[i][["text_raw", "sentiment_score"]].head(20))


#%%
# 일별 통계량
import matplotlib.pyplot as plt

for y in years:
    df = dfs[y].copy()
    
    df["date"] = pd.to_datetime(df["comment_at"]).dt.date
    
    daily_stats = df.groupby("date")["sentiment_score"].agg(
        mean="mean",
        std="std",
        count="count"
    )
    
    print(f"\n===== {y} =====")
    print(daily_stats.describe())

#%%
# 일별 평균 감성 점수 시각화
for y in years:
    df = dfs[y].copy()
    df["date"] = pd.to_datetime(df["comment_at"]).dt.date
    
    daily_mean = df.groupby("date")["sentiment_score"].mean()
    
    plt.figure(figsize=(10,4))
    plt.plot(daily_mean.index, daily_mean.values)
    plt.axhline(0, linestyle="--")
    plt.title(f"Daily Mean Sentiment - {y}")
    plt.xticks(rotation=45)
    plt.show()

#%%
# 일별 평균 감성 점수 분포 비교
plt.figure(figsize=(8,6))

for y in years:
    df = dfs[y].copy()
    df["date"] = pd.to_datetime(df["comment_at"]).dt.date
    
    daily_mean = df.groupby("date")["sentiment_score"].mean()
    
    plt.hist(daily_mean, bins=30, alpha=0.4, label=str(y))

plt.axvline(0, linestyle="--")
plt.title("Distribution of Daily Mean Sentiment")
plt.legend()
plt.show()