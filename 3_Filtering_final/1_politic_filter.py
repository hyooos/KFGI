import pandas as pd
import numpy as np
import re
from collections import Counter

print("=" * 80)
print("정치 키워드 기반 댓글 분류 분석")
print("=" * 80)
year = 2023  # 데이터 연도에 맞게 수정
# ========== 1. 데이터 로딩 ==========
print("\n1. 데이터 로딩 중...")
df = pd.read_csv(f'/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_{year}_증시금리.csv')

# 빈 댓글 제거
df_original_len = len(df)
df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

print(f"   전체 댓글 수: {df_original_len:,}개")
print(f"   빈 댓글 제거 후: {len(df):,}개")
print(f"   제거된 댓글: {df_original_len - len(df):,}개")

# ========== 2. 정치 키워드 정의 ==========
print("\n2. 정치 키워드 패턴 정의")
print("-" * 80)

politics_patterns = [
    r'윤석열|석열|윤통|용산|이재명|재명|개딸|한동훈|동훈|뚜껑|조국|문재인|재앙|박근혜|근혜|이명박|MB',
    r'민주당|더불어민주당|국민의힘|국힘|국짐|정의당|개혁신당|조국혁신당|좌파|우파|좌빨|수구|빨갱이|종북|토착왜구',
    r'탄핵|계엄|특검|비상계엄|내란|반역|구속|체포|영장|기소|검찰|독재|관권선거|부정선거|공천',
    r'의원|국회의원|국회|여당|야당|거대야당|당대표|원내대표|장관|차관|국무총리|대통령실|방통위|권익위',
    r'친문|친명|친윤|비윤|반윤|개헌|정권교체|심판|지지자|촛불집회|태극기부대|정치인|정치질'
]

# 패턴별로 어떤 키워드가 포함되는지 출력
print("정의된 정치 키워드 패턴:")
for i, pattern in enumerate(politics_patterns, 1):
    keywords = pattern.split('|')
    print(f"  패턴 {i}: {', '.join(keywords)}")

# ========== 3. 분류 함수 정의 ==========
def contains_political_keywords(text):
    """
    텍스트에 정치 키워드가 포함되어 있는지 확인
    
    Parameters:
    -----------
    text : str
        검사할 댓글 텍스트
    
    Returns:
    --------
    bool : 정치 키워드 포함 여부
    """
    if pd.isna(text):
        return False
    
    text_lower = text.lower()
    
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def get_matched_keywords(text):
    """
    텍스트에서 매칭된 정치 키워드 목록 반환
    
    Parameters:
    -----------
    text : str
        검사할 댓글 텍스트
    
    Returns:
    --------
    list : 매칭된 키워드 리스트
    """
    if pd.isna(text):
        return []
    
    text_lower = text.lower()
    matched = []
    
    for pattern in politics_patterns:
        # 패턴에 매칭되는 키워드 찾기
        keywords = pattern.split('|')
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)
    
    return list(set(matched))  # 중복 제거

# ========== 4. 분류 실행 ==========
print("\n3. 정치 키워드 기반 댓글 분류 중...")

df['is_political'] = df['text_raw'].apply(contains_political_keywords)
df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
df['keyword_count'] = df['matched_keywords'].apply(len)

# 분류 결과
political_comments = df[df['is_political']].copy()
non_political_comments = df[~df['is_political']].copy()

print("\n분류 결과:")
print("-" * 80)
print(f"정치 관련 댓글: {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
print(f"비정치 댓글: {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")

# ========== 5. 매칭된 키워드 통계 ==========
print("\n4. 매칭된 정치 키워드 통계")
print("-" * 80)

# 모든 매칭된 키워드 수집
all_matched_keywords = []
for keywords in political_comments['matched_keywords']:
    all_matched_keywords.extend(keywords)

keyword_freq = Counter(all_matched_keywords)
print(f"\n가장 많이 매칭된 정치 키워드 TOP 20:")
for keyword, count in keyword_freq.most_common(20):
    pct = count / len(political_comments) * 100
    print(f"  {keyword:15s}: {count:5,}회 (정치 댓글의 {pct:5.1f}%)")

# ========== 6. 샘플 댓글 확인 ==========
print("\n5. 샘플 댓글 확인")
print("-" * 80)

print("\n[정치 관련 댓글 샘플 10개]")
political_sample = political_comments.sample(min(10, len(political_comments)), random_state=42)
for idx, row in political_sample.iterrows():
    text = row['text_raw'][:80].replace('\n', ' ')
    keywords = ', '.join(row['matched_keywords'])
    print(f"  [키워드: {keywords}]")
    print(f"  {text}...\n")

print("\n[비정치 댓글 샘플 10개]")
non_political_sample = non_political_comments.sample(min(10, len(non_political_comments)), random_state=42)
for idx, row in non_political_sample.iterrows():
    text = row['text_raw'][:80].replace('\n', ' ')
    print(f"  {text}...\n")

# ========== 7. 결과 저장 ==========
print("\n6. 분류 결과 저장 중...")

# 출력 디렉토리 생성
import os
output_dir = '/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/political_filter'
os.makedirs(output_dir, exist_ok=True)

# CSV 파일로 저장
output_csv = f'{output_dir}/comments_{year}_classified.csv'
df[['news_id', 'comment_id', 'text_raw', 'is_political', 'matched_keywords', 
    'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
    output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_{year}_classified.csv")

# 정치 댓글만 별도 저장
political_output_csv = f'{output_dir}/comments_{year}_political_only.csv'
political_comments[['news_id', 'comment_id', 'text_raw', 'matched_keywords', 
                   'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
    political_output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_{year}_political_only.csv")

# 비정치 댓글만 별도 저장
non_political_output_csv = f'{output_dir}/comments_{year}_non_political_only.csv'
non_political_comments[['news_id', 'comment_id', 'text_raw', 'like_count', 
                        'dislike_count', 'comment_at']].to_csv(
    non_political_output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_{year}_non_political_only.csv")

# 정치 댓글 삭제된 원본 형식 파일 (모든 컬럼 유지)
cleaned_output_csv = f'{output_dir}/comments_political_removed_{year}.csv'
non_political_comments.drop(columns=['is_political', 'matched_keywords', 'keyword_count'], 
                            errors='ignore').to_csv(
    cleaned_output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: comments_political_removed_{year}.csv (정치 댓글 제거됨)")

# 통계 요약
summary = {
    'total_comments': len(df),
    'political_comments': len(political_comments),
    'non_political_comments': len(non_political_comments),
    'political_ratio': len(political_comments) / len(df) * 100,
    'avg_likes_political': political_comments['like_count'].mean(),
    'avg_likes_non_political': non_political_comments['like_count'].mean(),
    'top_10_keywords': dict(keyword_freq.most_common(10)),
}

import json
with open(f'{output_dir}/classification_summary_{year}.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✓ 저장: classification_summary.json")

# ========== 8. 결과 요약 ==========
print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n저장 위치: {output_dir}")
print("\n생성된 파일:")
print("  [데이터]")
print("  classified_comments.csv - 분류 결과 전체 (is_political 컬럼 포함)")
print("  political_comments_only.csv - 정치 댓글만")
print("  non_political_comments_only.csv - 비정치 댓글만")
print("  comments_political_removed.csv - 정치 댓글 제거된 클린 데이터 ⭐")
print("  classification_summary.json - 통계 요약")

print("\n" + "=" * 80)
print("요약 통계")
print("=" * 80)
print(f"전체 댓글: {len(df):,}개")
print(f"정치 관련 (삭제됨): {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
print(f"비정치 (유지됨): {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")
print(f"\n삭제 비율: {len(political_comments)/len(df)*100:.1f}%")
print(f"유지 비율: {len(non_political_comments)/len(df)*100:.1f}%")
print(f"\n평균 좋아요:")
print(f"  정치 관련 (삭제됨): {political_comments['like_count'].mean():.2f}개")
print(f"  비정치 (유지됨): {non_political_comments['like_count'].mean():.2f}개")
print(f"\n가장 많이 매칭된 키워드 TOP 5:")
for keyword, count in keyword_freq.most_common(5):
    print(f"  {keyword}: {count:,}회")
print("\n⭐ comments_political_removed.csv 파일이 정치 댓글이 제거된 클린 데이터입니다!")
print("=" * 80)