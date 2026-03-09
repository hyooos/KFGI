#%%
# data/KFG/sub_index_*.csv 파일들을 로드하여 하나의 데이터프레임으로 병합 후 저장
import pandas as pd
import os
from glob import glob

# sub_index_1
df1 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index1_momentum.csv")
# 1️⃣ 날짜 컬럼 이름 변경
df1 = df1.rename(columns={
    "Unnamed: 0": "date",
    "Momentum_Score": "sub_index1"
})

# 2️⃣ 날짜 타입 통일
df1["date"] = pd.to_datetime(df1["date"])

# 3️⃣ 필요한 컬럼만 남기기
df1 = df1[["date", "sub_index1"]]

new_row = {'date':pd.to_datetime('2022-05-09'), 'sub_index1' : 26.231305470845367}

df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)
df1 = df1.sort_values("date").reset_index(drop=True)
df1.head()
df1.tail()

#%%
# sub_index_2
df2 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index2_strength.csv")
df2 = df2.rename(columns={
    "Date": "date",
    "Strength_Score": "sub_index2"
})
df2["date"] = pd.to_datetime(df2["date"])
df2 = df2[["date", "sub_index2"]]
df2.head()

#%%
# sub_index_3
df3 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index3_osc.csv")
df3 = df3.rename(columns={
    "osc_score_0_100": "sub_index3"
})
df3["date"] = pd.to_datetime(df3["date"])
df3 = df3[["date", "sub_index3"]]
df3.head()

#%%
# sub_index_4
df4 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index4_pcratio.csv")
df4 = df4.rename(columns={
    "score_pcr": "sub_index4"
})
df4["date"] = pd.to_datetime(df4["date"])
df4 = df4[["date", "sub_index4"]]
df4.head()

#%%
# sub_index_5,7
df5 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index5_7_final.csv")
df5.columns
df5 = df5.rename(columns={
    "Unnamed: 0": "date",
    "Score_Fear": "sub_index5",
    "Final_JunkIndex": "sub_index7"
})
df5["date"] = pd.to_datetime(df5["date"])
df5 = df5[["date", "sub_index5", "sub_index7"]]
df5.head()

#%%
# sub_index_6
df6 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/sub_index6_safedemand.csv")
df6 = df6.rename(columns={
    "score_safe_demand": "sub_index6"
})
df6["date"] = pd.to_datetime(df6["date"])
df6 = df6[["date", "sub_index6"]]
df6.head()

#%%
# 2022-2025 코스피 지수 데이터
import FinanceDataReader as fdr
import pandas as pd
import numpy as np

# 2022-01-01 ~ 2025-12-31
kospi = fdr.DataReader("KS11", "2022-01-01", "2025-12-31")

kospi.head()

kospi = kospi.reset_index()
kospi = kospi.rename(columns={"Date": "date", "Close": "kospi_close"})

# 코스피 종가 지수만 남기기
kospi = kospi[["date", "kospi_close"]]

# 로그변환 + 차분 -> 코스피 로그 수익률
kospi["log_return"] = np.log(kospi["kospi_close"]).diff()
kospi = kospi.dropna().reset_index(drop=True)
kospi.head()

#%%
'''
나중에
df_all = kospi.merge(sentiment_df, on="date", how="left")
df_all["log_return_t+1"] = df_all["log_return"].shift(-1)
'''

#%%
# 먼저 모든 sub_index를 하나로 합친 후
df_sub = df1.merge(df2, on="date", how="outer") \
            .merge(df3, on="date", how="outer") \
            .merge(df4, on="date", how="outer") \
            .merge(df5, on="date", how="outer") \
            .merge(df6, on="date", how="outer")

df_sub = df_sub.sort_values("date")
df_sub.head()
df_sub.tail()

#%%
df_all = kospi.merge(df_sub, on="date", how="left")
df_all["log_return_t+1"] = df_all["log_return"].shift(-1)
df_all.head()

#%%
# daily_sentiment과 병합
daily_sentiment = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/sentiment_final/daily_sentiment.csv")
daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
final_df = df_all.merge(daily_sentiment, on="date", how="left")
final_df.head()

#%%
# 결측 확인
first_valid = final_df.apply(lambda col: col.first_valid_index())

first_valid
#%%
nan_after_start = {}

for col in final_df.columns:
    first_idx = final_df[col].first_valid_index()
    
    if first_idx is not None:
        after = final_df.loc[first_idx:, col]
        has_nan = after.isna().any()
        nan_after_start[col] = has_nan

nan_after_start

#%%
problem_cols = []

for col in final_df.columns:
    first_idx = final_df[col].first_valid_index()
    if first_idx is not None:
        after = final_df.loc[first_idx:, col]
        if after.isna().any():
            problem_cols.append(col)

problem_cols

#%%
# sub_index1 결측 위치
final_df[final_df["sub_index1"].isna()][["date", "sub_index1"]]
# log_return_t+1 결측 위치
final_df[final_df["log_return_t+1"].isna()][["date", "log_return_t+1"]]

#%%
final_df = final_df.dropna(subset=["log_return_t+1"])
final_df.head()
#%%
final = final_df.iloc[37:,:].copy()
final.head()

#%%
new_order = [
    "date",
    "sub_index1",
    "sub_index2",
    "sub_index3",
    "sub_index4",
    "sub_index5",
    "sub_index6",
    "sub_index7",
    "sent_norm_w",
    "sent_strength_w",
    "sent_std",
    "neg_z",
    "effective_n",
    "heat",
    "kospi_close",
    "log_return",
    "log_return_t+1"
]

final = final[new_order]
final = final.reset_index(drop=True)
final.head()

#%%
final.to_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv", index=False)