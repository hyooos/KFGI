#%%
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


KEYWORDS = ["증시", "국내증시", "주식시장", "금리"]

SECTION_LIST_ENDPOINT = "https://news.naver.com/section/template/SECTION_ARTICLE_LIST_FOR_LATEST"
COMMENT_COUNT_ENDPOINT = "https://news.naver.com/section/template/NEWS_COMMENT_COUNT_LIST"
BREAKING_BASE = "https://news.naver.com/breakingnews/section/101"  # /{sid2}?date=YYYYMMDD


@dataclass
class Article:
    list_date: str       # 우리가 섹션 페이지를 본 날짜(루프 날짜)
    sid2: int
    url: str
    oid: str
    aid: str
    title: str
    keyword: str
    object_id: str       # news{oid},{aid}
    pub_date: str = ""   # 기사 실제 작성일(가능하면 파싱)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    })
    return s


def safe_sleep(base: float):
    time.sleep(base + random.uniform(0.0, base * 0.5))


def daterange_yyyymmdd(start_yyyymmdd: str, end_yyyymmdd: str) -> Iterable[str]:
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def first_matched_keyword(title: str) -> str:
    for k in KEYWORDS:
        if k in title:
            return k
    return ""


def extract_oid_aid(url: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"/article/(\d{3})/(\d+)", url)
    if m:
        return m.group(1), m.group(2)
    m1 = re.search(r"[?&]oid=(\d{3})", url)
    m2 = re.search(r"[?&]aid=(\d+)", url)
    if m1 and m2:
        return m1.group(1), m2.group(1)
    return None


def strip_jsonp(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\((\{.*\})\)\s*;?\s*$", text, flags=re.DOTALL)
    return m.group(1) if m else text


def yyyymmdd_from_timestr(s: str) -> str:
    """
    comment/regTime 예: 2025-01-02T23:38:23+0900
    기사 메타 예: 2025-01-02T09:10:00+0900 또는 2025.01.02.
    """
    if not s:
        return ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return ""


def make_comment_referer(article_url: str) -> str:
    """
    보통은 https://n.news.naver.com/article/comment/{oid}/{aid} 형태.
    그냥 article_url을 Referer로 둬도 동작하는 경우가 있지만,
    최대한 안전하게 comment 페이지로 맞춰줌.
    """
    oa = extract_oid_aid(article_url)
    if not oa:
        return article_url
    oid, aid = oa
    return f"https://n.news.naver.com/article/comment/{oid}/{aid}"


def get_article_published_yyyymmdd(session: requests.Session, article_url: str, sleep_sec: float) -> str:
    """
    기사 본문 HTML에서 published_time 메타 등을 찾아 실제 작성일 추출 (best-effort).
    실패하면 빈 문자열 반환.
    """
    try:
        r = session.get(article_url, timeout=20)
        if r.status_code != 200:
            return ""
        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and meta.get("content"):
            d = yyyymmdd_from_timestr(meta["content"])
            if d:
                safe_sleep(sleep_sec)
                return d

        meta2 = soup.find("meta", attrs={"name": "article:published_time"})
        if meta2 and meta2.get("content"):
            d = yyyymmdd_from_timestr(meta2["content"])
            if d:
                safe_sleep(sleep_sec)
                return d

        m = re.search(r"(\d{4}\.\d{2}\.\d{2})\.", html)
        if m:
            d = yyyymmdd_from_timestr(m.group(1))
            safe_sleep(sleep_sec)
            return d

        safe_sleep(sleep_sec)
        return ""
    except Exception:
        return ""


def flatten_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_strings(v)
    elif isinstance(obj, str):
        yield obj


def parse_articles_from_html(html: str) -> List[Tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/article/" not in href:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        if href.startswith("/"):
            url = "https://news.naver.com" + href
        else:
            url = href
        out.append((url, title))
    seen = set()
    uniq = []
    for u, t in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((u, t))
    return uniq


def get_initial_next_from_html(session: requests.Session, date: str, sid2: int, sleep_sec: float) -> Optional[str]:
    url = f"{BREAKING_BASE}/{sid2}"
    r = session.get(url, params={"date": date}, timeout=20)
    if r.status_code != 200:
        return None
    html = r.text
    m = re.search(r"SECTION_ARTICLE_LIST_FOR_LATEST[^\"']*next=(\d{12,})", html)
    safe_sleep(sleep_sec)
    return m.group(1) if m else None


def fetch_section_articles_for_day(session: requests.Session, date: str, sid2: int,
                                  sleep_sec: float, max_pages: int = 80) -> List[Tuple[str, str]]:
    all_items: List[Tuple[str, str]] = []
    seen = set()

    next_token = get_initial_next_from_html(session, date, sid2, sleep_sec=sleep_sec)

    for page_no in range(1, max_pages + 1):
        params = {
            "sid": "101",
            "sid2": str(sid2),
            "cluid": "",
            "pageNo": str(page_no),
            "date": date,
        }
        if next_token:
            params["next"] = next_token

        r = session.get(SECTION_LIST_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            break

        try:
            data = r.json()
        except Exception:
            try:
                data = json.loads(r.text)
            except Exception:
                break

        html_snips = [s for s in flatten_strings(data) if "/article/" in s]
        page_new = 0
        for snip in html_snips:
            for url, title in parse_articles_from_html(snip):
                if url in seen:
                    continue
                seen.add(url)
                all_items.append((url, title))
                page_new += 1

        found_next = None
        if isinstance(data, dict) and isinstance(data.get("next"), str):
            found_next = data["next"]
        if not found_next:
            m = re.search(rf"{date}\d{{6,}}", r.text)
            if m:
                found_next = m.group(0)
        if found_next:
            next_token = found_next

        if page_new == 0:
            break

        safe_sleep(sleep_sec)

    return all_items


def chunked(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def fetch_comment_counts(session: requests.Session, object_ids: List[str], sleep_sec: float,
                        chunk_size: int = 80) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for part in chunked(object_ids, chunk_size):
        params = {"ticket": "news", "objectIds": ";".join(part)}
        r = session.get(COMMENT_COUNT_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            safe_sleep(sleep_sec)
            continue
        try:
            data = r.json()
        except Exception:
            try:
                data = json.loads(r.text)
            except Exception:
                safe_sleep(sleep_sec)
                continue

        def walk(obj):
            if isinstance(obj, dict):
                if "objectId" in obj:
                    oid = obj.get("objectId")
                    for k in ("commentCount", "count", "totalCount"):
                        if k in obj:
                            try:
                                counts[str(oid)] = int(obj[k])
                            except Exception:
                                pass
                            break
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(data)
        safe_sleep(sleep_sec)

    return counts


def build_comment_list_url(template_url: str, object_id: str, sort: str,
                           page_size: int, page_no: int,
                           more_next: Optional[str]) -> str:
    """
    template_url(사용자가 캡처한 web_naver_list_jsonp URL)을 기반으로
    objectId/sort/pageSize/page/pageType/initialize/moreParam.next만 덮어씀.
    """
    u = urlparse(template_url)
    q = parse_qs(u.query)

    q["objectId"] = [object_id]
    q["sort"] = [sort]
    q["pageSize"] = [str(page_size)]
    q["pageType"] = ["more"]
    q["page"] = [str(page_no)]
    q["initialize"] = ["true" if page_no == 1 else "false"]

    # 페이지네이션 핵심: moreParam.next
    if more_next:
        q["moreParam.next"] = [more_next]
    else:
        q.pop("moreParam.next", None)

    new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=False)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def parse_comments_from_payload(data: dict, sort: str) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    """
    return: (comments, next_id, end_id)
    """
    comments: List[Dict] = []
    result = data.get("result", {}) if isinstance(data, dict) else {}
    more_page = result.get("morePage", {}) if isinstance(result, dict) else {}
    next_id = more_page.get("next") if isinstance(more_page, dict) else None
    end_id = more_page.get("end") if isinstance(more_page, dict) else None

    def walk(obj):
        if isinstance(obj, dict):
            if ("commentNo" in obj or "commentId" in obj) and ("contents" in obj or "content" in obj or "text" in obj):
                cid = obj.get("commentNo", obj.get("commentId"))
                text = obj.get("contents", obj.get("content", obj.get("text", "")))
                like_cnt = obj.get("sympathyCount", obj.get("likeCount", obj.get("goodCount", 0)))
                dislike_cnt = obj.get("antipathyCount", obj.get("dislikeCount", obj.get("badCount", 0)))
                created = obj.get("regTime", obj.get("createdAt", obj.get("time", "")))

                comments.append({
                    "comment_id": str(cid),
                    "comment_at": str(created),
                    "text_raw": str(text),
                    "like_count": int(like_cnt) if str(like_cnt).isdigit() else 0,
                    "dislike_count": int(dislike_cnt) if str(dislike_cnt).isdigit() else 0,
                    "sort": sort,
                })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(result)  # result 안에 commentList가 있으니 result부터 훑음
    return comments, next_id, end_id


def fetch_comments_page(session: requests.Session, article_url: str, template_url: str,
                        object_id: str, sort: str, page_size: int,
                        page_no: int, more_next: Optional[str],
                        sleep_sec: float) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    url = build_comment_list_url(template_url, object_id, sort, page_size, page_no, more_next)
    headers = {"Referer": make_comment_referer(article_url)}
    r = session.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return [], None, None

    body = strip_jsonp(r.text)
    try:
        data = json.loads(body)
    except Exception:
        return [], None, None

    comments, next_id, end_id = parse_comments_from_payload(data, sort=sort)
    safe_sleep(sleep_sec)
    return comments, next_id, end_id


def collect_same_day_comments_topliked(session: requests.Session, article_url: str, template_url: str,
                                       object_id: str, target_day: str,
                                       want_n: int,
                                       page_size: int,
                                       max_pages: int,
                                       sleep_sec: float) -> List[Dict]:
    """
    1) sort=old(과거순)로 페이지네이션( moreParam.next ) 하면서
    2) 댓글 작성일 == target_day 인 것만 모으고
    3) target_day를 넘어서는 날짜가 나오면 중단
    4) 모은 댓글 중 like_count 내림차순으로 want_n개 반환
    """
    collected: Dict[str, Dict] = {}
    page_no = 1
    more_next = None

    for _ in range(max_pages):
        items, next_id, end_id = fetch_comments_page(
            session=session,
            article_url=article_url,
            template_url=template_url,
            object_id=object_id,
            sort="old",
            page_size=page_size,
            page_no=page_no,
            more_next=more_next,
            sleep_sec=sleep_sec,
        )
        if not items:
            break

        stop = False
        for it in items:
            d = yyyymmdd_from_timestr(it.get("comment_at", ""))
            if not d:
                continue
            if d == target_day:
                collected[it["comment_id"]] = it
            elif d > target_day:
                stop = True
                break
            # d < target_day 는 (과거순이면) 보통 나오지 않지만, 나오면 무시

        if stop:
            break

        # 다음 페이지 토큰 갱신
        if not next_id or (end_id and next_id == end_id):
            break
        more_next = next_id
        page_no += 1

    # 그날 댓글 중 공감(좋아요) 기준 상위 want_n
    out = list(collected.values())
    out.sort(key=lambda x: (x.get("like_count", 0), x.get("comment_at", "")), reverse=True)
    return out[:want_n]


def ensure_csv(path: str, header: List[str]):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as _:
            return
    except FileNotFoundError:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header)


def append_rows(path: str, rows: List[List]):
    if not rows:
        return
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)