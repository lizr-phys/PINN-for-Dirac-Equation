"""Verify selected paper identifiers against Crossref and author arXiv pages.
No credentials; standard-library only. All raw successful responses and errors
are retained. This checks metadata, not research quality or performance.
"""
from __future__ import annotations
import concurrent.futures
import html
import json
import re
import time
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "QuantumPINN-LiteratureReview/1.0 (academic metadata verification; Python urllib)"}

def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=35) as response:
        return response.read().decode("utf-8")

def arxiv_metadata(page):
    tags = re.findall(r"<meta\b[^>]*>", page, flags=re.I)
    result = {}
    for tag in tags:
        attrs = dict(re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', tag))
        name = attrs.get("name", "")
        if name.startswith("citation_"):
            result.setdefault(name, []).append(html.unescape(attrs.get("content", "")))
    abstract = re.search(r'<blockquote[^>]*class=["\']abstract[^"\']*["\'][^>]*>(.*?)</blockquote>', page, flags=re.S)
    if abstract:
        text = re.sub("<[^>]+>", " ", abstract.group(1))
        result["abstract"] = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return result

def resolve(item):
    rec = dict(item, checked_at=datetime.now(timezone.utc).isoformat())
    if item["doi"]:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(item["doi"], safe="")
        try:
            saved = RAW / f"{item['id']}_crossref.json"
            raw = saved.read_text(encoding="utf-8") if saved.exists() else fetch(url)
            saved.write_text(raw, encoding="utf-8")
            msg = json.loads(raw)["message"]
            rec["crossref"] = {k: msg.get(k) for k in (
                "DOI","title","author","container-title","type","volume","issue",
                "page","article-number","published","published-online",
                "published-print","created","URL","abstract")}
            rec["doi_status"] = "metadata_verified"
        except Exception as exc:
            rec["doi_status"] = "manual_needed"
            rec["doi_error"] = f"{type(exc).__name__}: {exc}"
    if item["arxiv"]:
        try:
            saved = RAW / f"{item['id']}_arxiv.html"
            raw = saved.read_text(encoding="utf-8") if saved.exists() else fetch("https://arxiv.org/abs/" + item["arxiv"])
            saved.write_text(raw, encoding="utf-8")
            rec["arxiv_metadata"] = arxiv_metadata(raw)
            rec["arxiv_status"] = "primary_page_verified"
            if not rec["arxiv_metadata"].get("citation_title"):
                rec["arxiv_status"] = "manual_needed"
        except Exception as exc:
            rec["arxiv_status"] = "manual_needed"
            rec["arxiv_error"] = f"{type(exc).__name__}: {exc}"
    return rec

def main():
    start = time.perf_counter()
    items = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(resolve, items))
    out = {"checked_at":datetime.now(timezone.utc).isoformat(),
           "scope":"Selected primary sources; not an exhaustive systematic review",
           "elapsed_seconds":time.perf_counter()-start,"records":records}
    (ROOT / "verified_metadata.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    for rec in records:
        title = (rec.get("crossref", {}).get("title") or rec.get("arxiv_metadata", {}).get("citation_title") or ["UNKNOWN"])[0]
        print(rec["id"], rec.get("doi_status", ""), rec.get("arxiv_status", ""), title, flush=True)
if __name__ == "__main__":
    main()
