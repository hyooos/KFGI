#%%
# crawling_naver_2024.py

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils_crawling import *

KEYWORDS = ["증시", "국내증시", "주식시장", "금리"]

SECTION_LIST_ENDPOINT = "https://news.naver.com/section/template/SECTION_ARTICLE_LIST_FOR_LATEST"
COMMENT_COUNT_ENDPOINT = "https://news.naver.com/section/template/NEWS_COMMENT_COUNT_LIST"
BREAKING_BASE = "https://news.naver.com/breakingnews/section/101"  # /{sid2}?date=YYYYMMDD



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--topk", type=int, default=5)          # 섹션별 TopK
    ap.add_argument("--per_article", type=int, default=30)  # 기사당 저장할 댓글 수(그날+공감상위)
    ap.add_argument("--sleep", type=float, default=0.9)
    ap.add_argument("--comment_template_url", required=True)
    ap.add_argument("--out_news", default="data/NAVER/article/news_2024_top5.csv")
    ap.add_argument("--out_comments", default="data/NAVER/comments/comments_2024_top5.csv")
    ap.add_argument("--test_days", type=int, default=0, help="예: 3이면 3일만 테스트")
    ap.add_argument("--comment_page_size", type=int, default=100)
    ap.add_argument("--max_comment_pages", type=int, default=50)
    ap.add_argument("--strict_pubdate", action="store_true",
                    help="기사 실제 작성일(pub_date)이 루프 날짜와 다르면 해당 기사 스킵")
    args = ap.parse_args()

    session = make_session()

    ensure_csv(args.out_news, [
        "loop_date", "pub_date", "news_id", "section", "keyword", "title",
        "comment_total_all", "rank_in_section", "url"
    ])
    ensure_csv(args.out_comments, [
        "news_id", "pub_date", "comment_id", "comment_at",
        "text_raw", "like_count", "dislike_count"
    ])

    dates = list(daterange_yyyymmdd(args.start, args.end))
    if args.test_days > 0:
        dates = dates[:args.test_days]

    processed_news = set()

    for loop_date in tqdm(dates, desc="Dates"):
        for sid2 in (259, 258):  # 금융, 증권
            # 1) 섹션/날짜 기사 수집
            raw_items = fetch_section_articles_for_day(session, loop_date, sid2, sleep_sec=args.sleep)

            candidates: List[Article] = []
            for url, title in raw_items:
                kw = first_matched_keyword(title)
                if not kw:
                    continue
                oa = extract_oid_aid(url)
                if not oa:
                    continue
                oid, aid = oa
                obj_id = f"news{oid},{aid}"
                candidates.append(Article(
                    list_date=loop_date, sid2=sid2, url=url, oid=oid, aid=aid,
                    title=title, keyword=kw, object_id=obj_id
                ))

            if not candidates:
                continue

            # 2) 댓글 총개수(전체 누적)로 섹션별 TopK 선정
            obj_ids = list({a.object_id for a in candidates})
            counts = fetch_comment_counts(session, obj_ids, sleep_sec=args.sleep)

            scored = [(counts.get(a.object_id, 0), a) for a in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)

            top: List[Tuple[int, Article]] = []
            seen_in_section = set()
            for c, a in scored:
                news_id = f"{a.oid}_{a.aid}"
                if news_id in seen_in_section:
                    continue
                if news_id in processed_news:
                    continue

                 # 기사 작성일 파싱 (TopK 채우는 동안만 필요한 만큼 호출됨)
                pub_date = get_article_published_yyyymmdd(session, a.url, sleep_sec=args.sleep) or loop_date
                a.pub_date = pub_date

                if args.strict_pubdate and pub_date != loop_date:
                    # strict면 이 기사 자체를 Top 후보에서 제외하고 다음 후보로 채우기
                    continue

                seen_in_section.add(news_id)
                top.append((c, a))
                if len(top) >= args.topk:
                    break

            if not top:
                continue

            # 3) news 저장 + 4) 댓글 저장(기사 작성일과 같은 댓글만 → 그중 공감상위 N개)
            news_rows = []
            comment_rows = []

            for rank, (c_total, a) in enumerate(top, start=1):
                news_id = f"{a.oid}_{a.aid}"

                pub_date = a.pub_date or loop_date

                if args.strict_pubdate and pub_date != loop_date:
                    # 루프 날짜와 기사 작성일이 다르면 스킵(원하면 옵션으로 엄격하게)
                    continue

                processed_news.add(news_id)

                news_rows.append([
                    loop_date, pub_date, news_id, a.sid2, a.keyword, a.title,
                    c_total, rank, a.url
                ])

                day_comments = collect_same_day_comments_topliked(
                    session=session,
                    article_url=a.url,
                    template_url=args.comment_template_url,
                    object_id=a.object_id,
                    target_day=pub_date,
                    want_n=args.per_article,
                    page_size=args.comment_page_size,
                    max_pages=args.max_comment_pages,
                    sleep_sec=args.sleep,
                )

                for it in day_comments:
                    comment_rows.append([
                        news_id,
                        pub_date,
                        it["comment_id"],
                        it["comment_at"],
                        it["text_raw"],
                        it["like_count"],
                        it["dislike_count"],
                    ])

            append_rows(args.out_news, news_rows)
            append_rows(args.out_comments, comment_rows)
            safe_sleep(args.sleep)


if __name__ == "__main__":
    main()
