"""
🔍 Research Radar — BenderLabs
Kaynak: ArXiv API
LLM: Ollama (lokal, sıfır maliyet)

Kullanım:
    python radar.py                   # Son 7 gün
    python radar.py --days 14         # Son 14 gün
    python radar.py --max 30          # Max 30 makale analiz et
    python radar.py --no-llm          # Sadece keyword filter
    python radar.py --model deepseek-r1:8b
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:8b"
REPORTS_DIR   = Path("./reports")
ARXIV_API     = "http://export.arxiv.org/api/query"

ARXIV_QUERIES = [
    ("cat:q-fin.TR OR cat:q-fin.PM OR cat:q-fin.ST OR cat:q-fin.RM",
     "q-fin · Quantitative Finance", True),
    ("cat:cs.AI AND (ti:trading OR ti:financial OR ti:stock OR ti:portfolio OR ti:sentiment OR ti:forecasting)",
     "cs.AI · AI + Finance", False),
    ("cat:cs.LG AND (ti:trading OR ti:financial OR ti:stock OR ti:portfolio OR ti:forecasting OR ti:time+series)",
     "cs.LG · ML + Finance", False),
]

BROAD_KEYWORDS = [
    "stock", "equity", "trading", "portfolio", "financial", "market",
    "hedge", "asset", "return", "alpha", "factor",
    "earnings", "sec filing", "10-k", "10-q", "annual report",
    "sentiment", "news", "volatility", "price prediction",
    "backtest", "sharpe", "drawdown",
    "xgboost", "gradient boosting", "random forest", "lstm", "transformer",
    "reinforcement learning", "time series", "forecasting",
    "rag", "retrieval", "embedding",
    "interest rate", "inflation", "gdp", "federal reserve", "macro",
    "shap", "explainab", "interpretab", "feature importance",
    "llm agent", "multi-agent", "trading agent", "financial agent",
]

# BenderEdge'in tam bileşen listesi — Aşama 1 ve 2 için kullanılır
BENDER_COMPONENTS = """
BenderEdge codebase (Python/FastAPI backend):

agents/quant.py        — RSI(14), MACD, Bollinger Bands, SMA20/50, ATR, volume spike
agents/ml_agent.py     — XGBoost on-the-fly training, SHAP, multi-horizon (1-7d/1-3m/3m+),
                          sector-aware hyperparams, walk-forward backtest, SMOTE
agents/sentiment.py    — headline tone scoring (-10 to +10)
agents/researcher.py   — NewsAPI + LLM analysis, spotlight headlines
agents/insider.py      — SEC EDGAR Form 4 insider trades
agents/macro.py        — FRED API: fed funds rate, CPI, GDP, unemployment
agents/earnings.py     — EPS history, beat/miss streaks, surprise %
agents/portfolio.py    — weighted voting (Quant 30%, ML 25%, Sentiment 15%,
                          Research 10%, Earnings 10%, Macro 5%, Insider 5%)

BenderLabs research targets:
- Financial RAG Lab    — retrieval over SEC filings / earnings transcripts
- Agent Debate Lab     — bull vs bear agent debate before final verdict
- Feature Research Lab — new technical/fundamental features for XGBoost
- Explainability Lab   — better SHAP usage, decision explanations
- Backtest Validation  — prediction accuracy tracking vs real prices
- Memory Lab           — long-term company knowledge persistence
"""

SCORE_LABEL = {5: "critical", 4: "high", 3: "medium", 2: "low"}

# ─────────────────────────────────────────────
# ARXIV API
# ─────────────────────────────────────────────

def fetch_arxiv_api(query: str, days: int, max_results: int = 100) -> list[dict]:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    date_filter = f" AND submittedDate:[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"

    params = urllib.parse.urlencode({
        "search_query": query + date_filter,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchRadar/2.0 BenderLabs"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

    ns    = {"atom": "http://www.w3.org/2005/Atom"}
    root  = ET.fromstring(raw)
    items = []

    for entry in root.findall("atom:entry", ns):
        title   = (entry.findtext("atom:title",   "", ns) or "").strip().replace("\n", " ")
        link_el = entry.find("atom:id", ns)
        link    = (link_el.text or "").strip() if link_el is not None else ""
        summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:800]
        if title and link:
            items.append({"title": title, "link": link, "summary": summary})

    return items


def broad_filter(paper: dict) -> bool:
    haystack = (paper["title"] + " " + paper["summary"]).lower()
    return any(kw in haystack for kw in BROAD_KEYWORDS)


# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────

def ollama_call(prompt: str, model: str, max_tokens: int = 600) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1, "num_predict": max_tokens}
    }, timeout=180)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    # deepseek-r1 <think> bloklarını temizle
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def extract_json(raw: str) -> dict:
    """Ham metinden JSON çıkar — markdown fence ve think bloklarını temizler."""
    raw = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"JSON bulunamadı: {raw[:200]}")


# ─────────────────────────────────────────────
# AŞAMA 1 — ELEME
# ─────────────────────────────────────────────

STAGE1_PROMPT = """You are a strict research filter for BenderEdge, a financial multi-agent AI system.

{components}

TASK: Does this paper propose a CONCRETE, IMPLEMENTABLE technique that can be directly
added to one of the BenderEdge files listed above?

Strict criteria for YES:
1. The paper must propose a specific algorithm, model architecture, or method (not just analysis)
2. It must be applicable to finance/trading (not just general ML theory)
3. A developer could realistically implement it in BenderEdge within weeks

Say NO if:
- Paper is purely theoretical with no practical algorithm
- Paper analyzes existing methods without proposing improvements
- Domain is unrelated (insurance, pensions, general options pricing unrelated to equity signals)
- Contribution is only mathematical proofs with no implementable output

PAPER:
Title: {title}
Abstract: {summary}

Respond with ONLY this JSON (nothing else):
{{"applicable": true, "reason": "one specific sentence about what the implementable technique is"}}"""


def stage1(paper: dict, model: str) -> tuple[bool, str]:
    prompt = STAGE1_PROMPT.format(
        components=BENDER_COMPONENTS,
        title=paper["title"],
        summary=paper["summary"],
    )
    try:
        raw  = ollama_call(prompt, model, max_tokens=200)
        data = extract_json(raw)
        return bool(data.get("applicable", False)), data.get("reason", "")
    except Exception as e:
        return True, f"Parse error (letting through): {e}"


# ─────────────────────────────────────────────
# AŞAMA 2 — DERIN ANALİZ
# ─────────────────────────────────────────────

STAGE2_PROMPT = """You are a senior ML/quant engineer reviewing a paper for the BenderEdge codebase.

{components}

Analyze this paper and fill in the JSON template below.
Be SPECIFIC and HONEST. Do NOT write generic statements like "XGBoost accuracy +5%".
Instead write what THIS paper specifically offers.

PAPER:
Title: {title}
Abstract: {summary}
Link: {link}

Score guide (be strict — most papers should be 2 or 3):
5 = CRITICAL: Novel technique that directly solves a known BenderEdge weakness. Implement this week.
4 = HIGH: Clear improvement to a specific agent. Implement within a month.
3 = MEDIUM: Interesting idea but requires significant research before implementing.
2 = LOW: Tangentially related. Good to know but unlikely to be implemented soon.
1 = SKIP: Not applicable after closer reading.

Difficulty:
easy   = <1 day, few lines of code change
medium = 2-5 days, new function or class
hard   = 1+ weeks, architectural change

Respond with ONLY this JSON (nothing else, no markdown):
{{
  "summary": "2-3 sentences: what the paper does and what problem it solves",
  "technique": "The specific algorithm/method proposed. Be precise, name the method.",
  "how_to_implement": "Specific file in BenderEdge to modify, what to add/change. Name the function/class.",
  "expected_gain": "Specific expected improvement for BenderEdge. No generic percentages.",
  "difficulty": "easy or medium or hard",
  "modules": ["exact agent or lab name from BenderEdge"],
  "score": 3,
  "score_reason": "One honest sentence explaining why this specific score."
}}"""


def stage2(paper: dict, model: str) -> dict:
    prompt = STAGE2_PROMPT.format(
        components=BENDER_COMPONENTS,
        title=paper["title"],
        summary=paper["summary"],
        link=paper["link"],
    )
    try:
        raw  = ollama_call(prompt, model, max_tokens=600)
        data = extract_json(raw)
        # Zorunlu alanlar var mı kontrol et
        for key in ["summary", "technique", "how_to_implement", "score"]:
            if key not in data:
                raise ValueError(f"Eksik alan: {key}")
        return data
    except Exception as e:
        return {
            "summary": paper["summary"][:300],
            "technique": "Analysis failed.",
            "how_to_implement": str(e)[:200],
            "expected_gain": "—",
            "difficulty": "—",
            "modules": [],
            "score": 1,
            "score_reason": "Parse/analysis error.",
        }


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def analyze_all(papers: list[dict], model: str) -> tuple[list[dict], int]:
    results = []
    skipped = 0

    for i, paper in enumerate(papers, 1):
        print(f"\n  [{i:02d}/{len(papers)}] {paper['title'][:60]}...")

        applicable, reason = stage1(paper, model)

        if not applicable:
            skipped += 1
            print(f"          ✗ SKIP  — {reason}")
            continue

        print(f"          ✓ PASS  — {reason}")
        print(f"          → Deep analysis...")

        paper["analysis"] = stage2(paper, model)
        s   = paper["analysis"].get("score", "?")
        mod = ", ".join(paper["analysis"].get("modules", []))
        dif = paper["analysis"].get("difficulty", "—")
        print(f"          ✓ Score: {s}/5 · {dif} · {mod or '—'}")
        results.append(paper)

    return results, skipped


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def build_report(analyzed: list[dict], skipped: int,
                 days: int, total_fetched: int, total_filtered: int,
                 model: str) -> str:
    today   = datetime.now().strftime("%d %B %Y")
    date_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Skora göre sırala, düşük skorları filtrele (1 = gösterme)
    analyzed = [p for p in analyzed if p["analysis"].get("score", 1) > 1]
    analyzed.sort(key=lambda p: p["analysis"].get("score", 0), reverse=True)

    cnt = lambda s: sum(1 for p in analyzed if p["analysis"].get("score") == s)

    lines = [
        "# Research Radar",
        f"**Date:** {today}  ",
        f"**Model:** {model}  ",
        f"**Funnel:** {total_fetched} fetched → {total_filtered} keyword filter → "
        f"{len(analyzed)+skipped} Stage 1 → {len(analyzed)} analyzed",
        "", "---", "",
        "## Summary", "",
        "| Score | Count |",
        "|-------|-------|",
        f"| 5 — critical (implement this week) | {cnt(5)} |",
        f"| 4 — high (implement this month)    | {cnt(4)} |",
        f"| 3 — medium (worth researching)     | {cnt(3)} |",
        f"| 2 — low (good to know)             | {cnt(2)} |",
        "", "---", "",
        "## Papers", "",
    ]

    if not analyzed:
        lines += [
            "No applicable techniques found in this period.",
            "Try --days 14 for a wider search.",
            "", "---", "",
        ]
    else:
        for i, p in enumerate(analyzed, 1):
            a   = p["analysis"]
            s   = a.get("score", 2)
            mod = ", ".join(a.get("modules", [])) or "—"
            dif = a.get("difficulty", "—")

            lines += [
                f"### {i}. {p['title']}",
                f"{p['link']}  ",
                f"score: {s}/5 ({SCORE_LABEL.get(s, 'low')}) · difficulty: {dif} · modules: {mod}",
                f"_{a.get('score_reason', '')}_",
                "",
                "**What it does:**",
                a.get("summary", ""), "",
                "**The technique:**",
                a.get("technique", ""), "",
                "**How to implement:**",
                a.get("how_to_implement", ""), "",
                "**Expected gain:**",
                a.get("expected_gain", ""), "",
                "---", "",
            ]

    lines.append(f"*Research Radar · BenderLabs · {date_ts}*")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="🔍 Research Radar — BenderLabs")
    parser.add_argument("--days",   type=int, default=7,            help="Days to look back (default: 7)")
    parser.add_argument("--no-llm", action="store_true",            help="Skip LLM analysis")
    parser.add_argument("--max",    type=int, default=30,           help="Max papers to analyze (default: 30)")
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL,help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Research Radar  ·  {date_str}")
    print(f"  Model: {args.model}")
    print(f"{'─'*60}\n")

    # ── OLLAMA CHECK ────────────────────────────
    if not args.no_llm:
        try:
            r      = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"Ollama connected · Models: {', '.join(models)}\n")
            if not any(args.model in m for m in models):
                print(f"⚠️  {args.model} not found. Pulling...")
                import os; os.system(f"ollama pull {args.model}")
        except Exception as e:
            print(f"❌ Ollama error: {e}\n   → run: ollama serve")
            return

    # ── FETCH ───────────────────────────────────
    all_papers:   list[dict] = []
    total_fetched = 0
    print(f"Fetching last {args.days} days from ArXiv API...\n")

    for query, label, is_direct in ARXIV_QUERIES:
        print(f"  {label}")
        items    = fetch_arxiv_api(query, args.days, max_results=100)
        filtered = items if is_direct else [p for p in items if broad_filter(p)]
        total_fetched += len(items)
        print(f"    {len(items)} papers → filter: {len(filtered)}\n")
        all_papers.extend(filtered)
        time.sleep(1)

    # deduplicate
    seen, unique = set(), []
    for p in all_papers:
        if p["link"] not in seen:
            seen.add(p["link"])
            unique.append(p)
    all_papers     = unique
    total_filtered = len(all_papers)

    print(f"{total_filtered} relevant papers ({total_fetched} fetched)\n")

    if not all_papers:
        print("⚠️  No papers found. Try --days 14")
        return

    to_process = all_papers[:args.max]

    # ── ANALYSIS ────────────────────────────────
    analyzed_papers = []
    skipped_count   = 0

    if not args.no_llm:
        print(f"2-Stage Analysis · {len(to_process)} papers\n")
        print(f"  Stage 1: implementable technique? (strict filter)")
        print(f"  Stage 2: deep technical analysis (only papers that pass)\n")
        analyzed_papers, skipped_count = analyze_all(to_process, args.model)
    else:
        print("⚡ LLM skipped (--no-llm)\n")
        analyzed_papers = to_process

    # ── REPORT ──────────────────────────────────
    report = build_report(
        analyzed_papers, skipped_count,
        args.days, total_fetched, total_filtered, args.model
    )
    out = REPORTS_DIR / f"radar_{date_str}.md"
    out.write_text(report, encoding="utf-8")

    passed = len([p for p in analyzed_papers if p.get("analysis", {}).get("score", 0) > 1])
    print(f"\n{'─'*60}")
    print(f"  {passed} papers in report · {skipped_count} eliminated")
    print(f"  {out}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()