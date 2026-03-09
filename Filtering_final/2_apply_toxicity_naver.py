# apply_toxicity_navert.py
# Python 3.9 compatible

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


@torch.no_grad()
def predict_toxicity_binary(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: Optional[str] = None,
) -> List[float]:
    """
    Binary toxicity scoring:
    - expects 2-class classifier (softmax) where LABEL_1 is toxic/hate (common in many binary models)
    - returns toxicity score in [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    scores: List[float] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity"):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits  # (B, C)
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            p = probs[:, 1]  # assume index 1 is toxic

        scores.extend(p.detach().cpu().tolist())

    return scores


def summarize_thresholds(df: pd.DataFrame, mode: str, tau: float, hard_tau: float) -> None:
    total = len(df)
    empty = int(df["is_empty"].sum())
    non_empty = total - empty

    print("\n==================== SUMMARY ====================")
    print(f"total rows: {total}")
    print(f"empty text_raw: {empty}  | non-empty: {non_empty}")
    print(f"mode: {mode}")

    if mode == "drop":
        dropped = int((~df["keep"]).sum())
        kept = int(df["keep"].sum())
        print(f"drop rule: empty OR toxicity >= {tau}")
        print(f"kept: {kept} | dropped: {dropped}")
    else:
        hard_dropped = int((~df["keep"]).sum())
        print(f"weight rule: weight=(1-tox)^gamma, gamma applied")
        if hard_tau <= 1.0:
            print(f"hard drop rule: empty OR toxicity >= {hard_tau}")
            print(f"hard dropped(keep=False): {hard_dropped} (still saved in CSV, but flagged)")
        else:
            print("hard drop disabled (keep=True for all non-empty)")
    print("=================================================\n")


def print_examples(df: pd.DataFrame, title: str, sub: pd.DataFrame, show: int = 10) -> None:
    print(f"\n[{title}] (showing up to {show})")
    cols = ["toxicity_score", "weight", "keep", "is_empty", "like_count", "dislike_count", "text_raw"]
    cols = [c for c in cols if c in sub.columns]
    view = sub[cols].head(show).copy()

    # shorten text for console
    def _shorten(s: str, n: int = 120) -> str:
        s = str(s).replace("\n", " ")
        return s if len(s) <= n else s[:n] + "..."

    if "text_raw" in view.columns:
        view["text_raw"] = view["text_raw"].map(_shorten)

    # prettier printing
    with pd.option_context("display.max_colwidth", 200, "display.width", 120):
        print(view.to_string(index=False))


def print_bins(df: pd.DataFrame) -> None:
    """
    Show distribution by toxicity bins, and average weight per bin.
    """
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    labels = ["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,.9)", "[.9,.95)", "[.95,1]"]

    tmp = df[~df["is_empty"]].copy()
    if len(tmp) == 0:
        print("\n[BINS] no non-empty rows.")
        return

    tmp["tox_bin"] = pd.cut(tmp["toxicity_score"], bins=bins, labels=labels, include_lowest=True, right=False)

    g = tmp.groupby("tox_bin", dropna=False).agg(
        n=("toxicity_score", "size"),
        mean_tox=("toxicity_score", "mean"),
        mean_weight=("weight", "mean") if "weight" in tmp.columns else ("toxicity_score", "mean"),
        kept=("keep", "sum") if "keep" in tmp.columns else ("toxicity_score", "size"),
    ).reset_index()

    print("\n[TOXICITY BINS]")
    with pd.option_context("display.width", 120):
        print(g.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="comments_2024.csv")
    ap.add_argument("--out", default="comments_2024_100_toxicity.csv")
    ap.add_argument("--n", type=int, default=-1, help="how many rows; -1 means ALL")


    ap.add_argument("--model", default="jinkyeongk/kcELECTRA-toxic-detector")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--mode", default="weight", choices=["drop", "weight"])
    ap.add_argument("--tau", type=float, default=0.90)

    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--hard_tau", type=float, default=0.95)

    # reporting options
    ap.add_argument("--show", type=int, default=10, help="how many examples to print per group")
    ap.add_argument("--save_kept", default="comments_2024_100_kept.csv", help="save keep==True rows")
    ap.add_argument("--save_dropped", default="comments_2024_100_dropped.csv", help="save keep==False rows")

    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)

    required = ["news_id", "comment_id", "comment_at", "text_raw", "like_count", "dislike_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nColumns: {list(df.columns)}")

    if args.n is not None and args.n > 0:
        df = df.head(args.n).copy()
    else:
        df = df.copy()


    # empty handling
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df["is_empty"] = df["text_raw"].str.strip().eq("")

    # score only non-empty
    idx = df.index[~df["is_empty"]].tolist()
    texts = df.loc[idx, "text_raw"].tolist()

    df["toxicity_score"] = 0.0
    if len(texts) > 0:
        scores = predict_toxicity_binary(
            texts=texts,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        df.loc[idx, "toxicity_score"] = scores

    df["toxicity_score"] = df["toxicity_score"].fillna(0.0).clip(0.0, 1.0)

    if args.mode == "drop":
        df["weight"] = 1.0  # drop mode에서는 weight가 크게 의미 없어서 1로 둠(원하면 제거 가능)
        df["keep"] = (~df["is_empty"]) & (df["toxicity_score"] < args.tau)
        out_df = df[df["keep"]].copy()
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

        print(f"[SAVE] scored+filtered kept-only -> {args.out}")

    else:
        df["weight"] = (1.0 - df["toxicity_score"]).clip(lower=0.0) ** args.gamma

        if args.hard_tau is not None and args.hard_tau <= 1.0:
            df["keep"] = (~df["is_empty"]) & (df["toxicity_score"] < args.hard_tau)
        else:
            df["keep"] = ~df["is_empty"]

        df.loc[df["is_empty"], "weight"] = 0.0
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[SAVE] scored + keep/weight flags -> {args.out}")

    # save kept/dropped splits for inspection
    kept_df = df[df["keep"]].copy()
    dropped_df = df[~df["keep"]].copy()

    kept_df.to_csv(args.save_kept, index=False, encoding="utf-8-sig")
    dropped_df.to_csv(args.save_dropped, index=False, encoding="utf-8-sig")

    print(f"[SAVE] kept rows -> {args.save_kept}")
    print(f"[SAVE] dropped rows -> {args.save_dropped}")

    # reporting
    summarize_thresholds(df, args.mode, args.tau, args.hard_tau)
    print_bins(df)

    # examples
    print_examples(df, "MOST TOXIC", df.sort_values("toxicity_score", ascending=False), show=args.show)
    print_examples(df, "LEAST TOXIC (non-empty)", df[~df["is_empty"]].sort_values("toxicity_score"), show=args.show)
    print_examples(df, "DROPPED (keep=False)", dropped_df.sort_values("toxicity_score", ascending=False), show=args.show)
    print_examples(df, "KEPT (keep=True)", kept_df.sort_values("toxicity_score", ascending=False), show=args.show)


if __name__ == "__main__":
    main()
