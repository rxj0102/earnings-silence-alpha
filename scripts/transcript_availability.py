"""Measure how many earnings 8-K filings on EDGAR contain a call transcript, for the company universe.

For every 8-K with Item 2.02 (results of operations) filed 2015-01-01 to 2023-12-31, each EX-99 exhibit is
downloaded and classified with the same word-count rule as src/transcript_scraper.py:
transcript (> 2000 words and contains both Operator and question), press release (500 to 2000 words or no
call markers), insufficient (< 500 words).
Output: results/transcript_availability.csv (one row per earnings 8-K) and a printed summary.

Usage: EDGAR_USER_AGENT="Name email" python scripts/transcript_availability.py
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": os.getenv("EDGAR_USER_AGENT", "earnings-silence-alpha research")}
CACHE = ROOT / "data" / "raw" / "edgar_cache"


def get(url, as_json=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / re.sub(r"[^A-Za-z0-9]+", "_", url)[-180:]
    if key.exists():
        raw = key.read_text(errors="replace")
    else:
        for a in range(4):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
                break
            except Exception:
                if a == 3:
                    raise
                time.sleep(2 ** a)
        time.sleep(0.12)
        key.write_text(raw)
    return json.loads(raw) if as_json else raw


def classify(html):
    text = re.sub(r"<[^>]+>", " ", html)
    words = len(text.split())
    low = text.lower()
    if words < 500:
        return "insufficient", words
    if words > 2000 and "operator" in low and "question" in low:
        return "transcript", words
    return "press_release", words


def main():
    uni = pd.read_csv(ROOT / "data" / "reference" / "company_universe.csv", comment="#")
    rows = []
    for _, c in uni.iterrows():
        cik = str(c.cik).zfill(10)
        sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True)
        blocks = [sub["filings"]["recent"]] + [get(f"https://data.sec.gov/submissions/{f['name']}", as_json=True) for f in sub["filings"].get("files", [])]
        df = pd.concat([pd.DataFrame({k: b[k] for k in ("form", "filingDate", "accessionNumber", "items")}) for b in blocks])
        df = df[(df.form == "8-K") & df["items"].str.contains("2.02") & df.filingDate.between("2015-01-01", "2023-12-31")]
        for _, f in df.iterrows():
            nodash = f.accessionNumber.replace("-", "")
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/index.json", as_json=True)
            exhibits = [i["name"] for i in idx["directory"]["item"] if re.search(r"ex[-_]?99|exhibit99|ex99", i["name"], re.I) and i["name"].lower().endswith((".htm", ".html", ".txt"))]
            best, words = "no_exhibit", 0
            for e in exhibits:
                kind, w = classify(get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{e}"))
                if kind == "transcript" or (best != "transcript" and w > words):
                    best, words = kind, w
                if kind == "transcript":
                    break
            rows.append({"ticker": c.ticker, "filing_date": f.filingDate, "accession": f.accessionNumber, "best_exhibit_class": best, "words": words})
        sub_df = pd.DataFrame([r for r in rows if r["ticker"] == c.ticker])
        print(f"{c.ticker:6s} earnings 8-Ks {len(sub_df):3d}  transcripts {int((sub_df.best_exhibit_class == 'transcript').sum()) if len(sub_df) else 0}", flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "transcript_availability.csv", index=False)
    print(out.best_exhibit_class.value_counts().to_string())


if __name__ == "__main__":
    sys.exit(main())
