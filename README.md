# 🔍 Research Radar

> Daily AI/Finance research tracker — built for the [BenderEdge](https://github.com/yusufbender/BenderEdge) ecosystem.

Automatically fetches papers from **ArXiv** (q-fin + cs.AI/LG), runs a **2-stage local LLM filter** to find papers with implementable techniques, and generates a structured **Markdown report** — no API costs, no rate limits.

```
ArXiv API (q-fin + cs.AI/LG)
        │
        ▼
Keyword Pre-filter
        │
        ▼
Stage 1 — Ollama (strict): "Is there a concrete, implementable technique?"
        │
   ✗ Eliminated (most papers)
        │
        ▼
Stage 2 — Ollama (deep): Full analysis → score, difficulty, how to implement
        │
        ▼
reports/radar_YYYY-MM-DD.md
```

## Example Output

```
### Volatility Forecasting and Return Prediction under Market Regimes
Score: ⭐⭐⭐ (3/5) · Difficulty: 🟡 medium · Modules: ml_agent, quant

The technique:
Markov-switching GJR-GARCH for volatility modeling combined with XGBoost for return prediction.

How to implement in BenderEdge:
Modify agents/ml_agent.py to incorporate regime indicators and GJR-GARCH volatility
forecasts into the XGBoost model. Create a new class in agents/quant.py for regime detection.
```

See [examples/](examples/) for a full sample report.

## Setup

**Requirements:** Python 3.10+ · [Ollama](https://ollama.com)

```bash
git clone https://github.com/yusufbender/research-radar
cd research-radar
pip install requests
```

Pull a model (choose one):
```bash
ollama pull deepseek-r1:8b   # recommended — better reasoning
ollama pull qwen2.5:7b       # faster, lighter
```

## Usage

```bash
# Last 7 days (default)
python radar.py

# Last 14 days
python radar.py --days 14

# Analyze up to 50 papers
python radar.py --max 50

# Use a different model
python radar.py --model qwen2.5:7b

# Skip LLM — keyword filter only (instant)
python radar.py --no-llm
```

Reports are saved to `./reports/radar_YYYY-MM-DD.md`.

## How the filter works

**Stage 1 (fast):** For each paper, the LLM answers: *"Does this paper propose a concrete, implementable algorithm that can be directly added to BenderEdge?"* — strict YES/NO. Most papers are eliminated here.

**Stage 2 (deep):** For papers that pass, the LLM produces:
- What the paper does (2-3 sentences)
- The specific technique/algorithm
- Which BenderEdge file to modify and how
- Expected gain (specific, not generic)
- Difficulty: `easy` / `medium` / `hard`
- Score 1–5 (most papers score 2–3; 5 = implement this week)

## Customize for your own project

Edit `BENDER_COMPONENTS` in `radar.py` to describe your own codebase. The LLM will filter and analyze papers against your specific files and modules instead.

## Sources

| Feed | Coverage |
|------|----------|
| ArXiv `q-fin.TR` | Trading & Market Microstructure |
| ArXiv `q-fin.PM` | Portfolio Management |
| ArXiv `q-fin.ST` | Statistical Finance |
| ArXiv `q-fin.RM` | Risk Management |
| ArXiv `cs.AI` | AI papers mentioning finance/trading |
| ArXiv `cs.LG` | ML papers mentioning finance/trading |

## Part of BenderEdge Ecosystem

```
BenderLabs
├── 🔍 Research Radar   ← you are here
├── 🧠 BenderEdge       — multi-agent stock research platform
└── 📈 BenderQuant      — XGBoost financial ML engine
```

---

*Built by [Yusuf Bender](https://github.com/yusufbender) · Not financial advice*