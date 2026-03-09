import pandas as pd
import re
from collections import Counter
import numpy as np
import json

print("=" * 80)
print("주식/경제 키워드 분석 + 자동 필터링")
print("=" * 80)
year = 2025  # 데이터 연도에 맞게 수정
# ========== 1. 데이터 로딩 ==========
print("\n1. 데이터 로딩 중...")
input_file = f'/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/toxicity/comments_toxicity_kept_{year}.csv'
df = pd.read_csv(input_file)

# 빈 댓글 제거
df_original_len = len(df)
df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

print(f"   원본 댓글 수: {df_original_len:,}개")
print(f"   빈 댓글 제거 후: {len(df):,}개")

# ========== 2. 키워드 추출 및 분석 ==========
print("\n2. 주식/경제 키워드 자동 추출 중...")
print("-" * 80)

# 형태소 단위로 분리하지 않고, 2-6글자 단어 추출
def extract_words(text):
    if pd.isna(text):
        return []
    words = re.findall(r'[가-힣]{2,6}|[a-zA-Z]{2,10}', text.lower())
    return words

# 모든 댓글에서 단어 추출
all_words = []
for text in df['text_raw']:
    all_words.extend(extract_words(text))

word_freq = Counter(all_words)

print(f"총 추출 단어 수: {len(all_words):,}개")
print(f"유니크 단어 수: {len(word_freq):,}개")

# 주식/경제 관련 단어 패턴
stock_patterns = [
    r'코스피|코스닥|kospi|kosdaq',
    r'주식|주가|증시|시장|장',
    r'매수|매도|투자|손절|익절',
    r'개미|외인|기관|외국인',
    r'삼전|삼성전자|하닉|하이닉스',
    r'지수|시총|배당|상장',
    r'급등|급락|폭등|폭락|상승|하락',
    r'수급|거래량|환율|금리',
    r'종목|실적|반도체|전지',
    r'펀드|연기금|국민연금',
]

def is_stock_related_word(word):
    return any(re.search(pattern, word) for pattern in stock_patterns)

# 문서 빈도(DF) 계산
word_document_count = {}
for text in df['text_raw']:
    words = set(extract_words(text))
    for word in words:
        word_document_count[word] = word_document_count.get(word, 0) + 1

word_df = [(word, count) for word, count in word_document_count.items()]
word_df.sort(key=lambda x: x[1], reverse=True)

# 키워드 추출
threshold = len(df) * 0.01  # 1% 이상
core_keywords = []
support_keywords = []

for word, doc_count in word_df:
    if is_stock_related_word(word):
        pct = doc_count / len(df) * 100
        if doc_count >= threshold:
            core_keywords.append((word, doc_count, pct))
        elif doc_count >= threshold * 0.3:  # 0.3% 이상
            support_keywords.append((word, doc_count, pct))

print(f"\n자동 추출 결과:")
print(f"  핵심 키워드: {len(core_keywords)}개 (댓글의 1% 이상 출현)")
print(f"  보조 키워드: {len(support_keywords)}개 (댓글의 0.3-1% 출현)")

# 추출된 키워드를 리스트로 변환
CORE_STOCK_KEYWORDS = [word for word, _, _ in core_keywords]
SUPPORT_STOCK_KEYWORDS = [word for word, _, _ in support_keywords]

print(f"\n핵심 키워드 목록:")
for word, _, pct in core_keywords:
    print(f"  '{word}' ({pct:.1f}%)")

# ========== 3. 점수 계산 및 분류 ==========
print("\n3. 점수 계산 및 댓글 분류 중...")
print("-" * 80)

def score_comment(text):
    """
    댓글의 주식 관련도 점수 계산
    
    점수 = (핵심 키워드 개수 × 10) + (보조 키워드 개수 × 3)
    """
    if pd.isna(text):
        return 0
    
    text_lower = text.lower()
    score = 0
    
    # 핵심 주식 키워드 (높은 가중치)
    core_count = sum(1 for kw in CORE_STOCK_KEYWORDS if kw in text_lower)
    score += core_count * 10
    
    # 보조 주식 키워드 (낮은 가중치)
    support_count = sum(1 for kw in SUPPORT_STOCK_KEYWORDS if kw.lower() in text_lower)
    score += support_count * 3
    
    return score

def classify_comment(text, score):
    """
    댓글 분류
    
    조건:
    1. 점수 10점 이상
    2. 핵심 키워드 최소 1개 포함
    """
    if score < 10:
        return 'other'
    
    text_lower = text.lower() if pd.notna(text) else ''
    has_core = any(kw in text_lower for kw in CORE_STOCK_KEYWORDS)
    
    if not has_core:
        return 'other'
    
    return 'stock'

# 점수 계산 및 분류
df['stock_score'] = df['text_raw'].apply(score_comment)
df['is_stock'] = df.apply(lambda row: classify_comment(row['text_raw'], row['stock_score']), axis=1)

stock_df = df[df['is_stock'] == 'stock'].copy()
other_df = df[df['is_stock'] == 'other'].copy()

print(f"\n분류 결과:")
print(f"  주식/경제 관련: {len(stock_df):,}개 ({len(stock_df)/len(df)*100:.1f}%)")
print(f"  기타: {len(other_df):,}개 ({len(other_df)/len(df)*100:.1f}%)")

# 점수 분포
print(f"\n주식 관련 댓글 점수 분포:")
print(f"  평균 점수: {stock_df['stock_score'].mean():.1f}")
print(f"  중간 점수: {stock_df['stock_score'].median():.1f}")
print(f"  최고 점수: {stock_df['stock_score'].max()}")
print(f"  최저 점수: {stock_df['stock_score'].min()}")

# ========== 4. 샘플 확인 ==========
print("\n4. 샘플 댓글 확인")
print("-" * 80)

print("\n[주식/경제 관련 댓글 샘플 (점수 높은 순 10개)]")
top_stock = stock_df.nlargest(10, 'stock_score')
for idx, row in top_stock.iterrows():
    text = row['text_raw'][:100].replace('\n', ' ')
    print(f"[점수:{row['stock_score']:3d}] {text}...")

print("\n[기타 댓글 샘플 10개]")
sample_other = other_df.sample(min(10, len(other_df)), random_state=42)
for idx, row in sample_other.iterrows():
    text = row['text_raw'][:100].replace('\n', ' ')
    print(f"[점수:{row['stock_score']:3d}] {text}...")

# ========== 5. 결과 저장 ==========
print("\n5. 결과 저장 중...")
print("-" * 80)

import os
output_dir = '/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/final_filtered' # 최종 결과 저장 디렉토리(개인 환경에 맞게 수정)
os.makedirs(output_dir, exist_ok=True)

# (1) 키워드 분석 결과 JSON
keyword_result = {
    'core_keywords': [{'word': w, 'count': c, 'percentage': p} for w, c, p in core_keywords],
    'support_keywords': [{'word': w, 'count': c, 'percentage': p} for w, c, p in support_keywords],
}
with open(f'{output_dir}/keyword_analysis_result_{year}.json', 'w', encoding='utf-8') as f:
    json.dump(keyword_result, f, ensure_ascii=False, indent=2)
print(f"✓ 저장: keyword_analysis_result_2025.json")

# (2) 전체 데이터 (점수 + 분류 컬럼 포함)
df.to_csv(f'{output_dir}/classified_stock_comments_{year}.csv', index=False, encoding='utf-8-sig')
print(f"✓ 저장: classified_stock_comments_{year}.csv")

# (3) 주식/경제 관련 댓글만
stock_df.to_csv(f'{output_dir}/comments_final_stock_only_{year}.csv', index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_final_stock_only_{year}.csv")

# (4) 주식 관련 댓글 (분류 컬럼 제거 - 최종 클린 데이터)
stock_df.drop(columns=['stock_score', 'is_stock'], errors='ignore').to_csv(
    f'{output_dir}/comments_stock_clean_{year}.csv', index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_stock_clean_{year}.csv (분류 컬럼 제거)")

# (5) 기타 댓글 (참고용)
other_df.to_csv(f'{output_dir}/comments_other_{year}.csv', index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_other_{year}.csv")

# (6) 통계 요약
summary = {
    'total_comments': len(df),
    'stock_comments': len(stock_df),
    'other_comments': len(other_df),
    'stock_ratio': len(stock_df) / len(df) * 100,
    'avg_score': float(stock_df['stock_score'].mean()),
    'median_score': float(stock_df['stock_score'].median()),
    'core_keywords_count': len(core_keywords),
    'support_keywords_count': len(support_keywords),
}
with open(f'{output_dir}/stock_classification_summary_{year}.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✓ 저장: stock_classification_summary_{year}.json")

# ========== 6. 최종 요약 ==========
print("\n" + "=" * 80)
print("분석 및 필터링 완료!")
print("=" * 80)
print(f"\n저장 위치: {output_dir}")
print("\n생성된 파일:")
print("  [분석 결과]")
print(f"  keyword_analysis_result_{year}.json - 추출된 키워드 목록")
print(f"  stock_classification_summary_{year}.json - 통계 요약")
print("\n  [전체 데이터]")
print(f"  classified_stock_comments_{year}.csv - 전체 (점수/분류 컬럼 포함)")
print("\n  [주식 관련 댓글만]")
print(f"  comments_final_stock_only_{year}.csv - 주식 관련 (모든 컬럼)")
print(f"  comments_stock_clean_{year}.csv - 주식 관련 (분류 컬럼 제거) ⭐")
print("\n  [기타]")
print(f"  comments_other_{year}.csv - 기타 댓글")

print("\n" + "=" * 80)
print("요약 통계")
print("=" * 80)
print(f"입력: {input_file}")
print(f"전체 댓글: {len(df):,}개")
print(f"주식/경제: {len(stock_df):,}개 ({len(stock_df)/len(df)*100:.1f}%)")
print(f"기타: {len(other_df):,}개 ({len(other_df)/len(df)*100:.1f}%)")
print(f"\n핵심 키워드: {len(core_keywords)}개")
print(f"보조 키워드: {len(support_keywords)}개")
print(f"평균 점수: {stock_df['stock_score'].mean():.2f}")
print(f"\n⭐ comments_stock_clean_{year}.csv = 최종 클린 데이터")
print("   (정치 제거 → 유해성 제거 → 주식/경제만 추출)")
print("=" * 80)