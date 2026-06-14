# Research Radar

Daily AI/Finance paper tracker — built for the [BenderEdge](https://github.com/yusufbender/BenderEdge) ecosystem.

Fetches papers from ArXiv (q-fin + cs.AI/LG), runs a 2-stage local LLM filter to find papers with implementable techniques, and writes a Markdown report. No API costs, no rate limits.

```
ArXiv API (q-fin + cs.AI/LG)
        │
        ▼
Keyword pre-filter
        │
        ▼
Stage 1 — Ollama: "Is there a concrete, implementable technique?"
        │
   eliminated (most papers)
        │
        ▼
Stage 2 — Ollama: full analysis → score, difficulty, how to implement
        │
        ▼
reports/radar_YYYY-MM-DD.md
```

## Example output

```
### Volatility Forecasting and Return Prediction under Market Regimes
http://arxiv.org/abs/2606.09478v1
score: 3/5 (medium) · difficulty: medium · modules: ml_agent, quant
_Improves return prediction but requires significant integration with existing XGBoost framework._

**The technique:**
Markov-switching GJR-GARCH for volatility modeling combined with XGBoost for return prediction.

**How to implement:**
Modify agents/ml_agent.py to incorporate regime indicators and GJR-GARCH volatility
forecasts into the XGBoost model. Create a new class in agents/quant.py for regime detection.
```

See [examples/](examples/) for a full report.

## Setup

Requirements: Python 3.10+ · [Ollama](https://ollama.com)

```bash
git clone https://github.com/yusufbender/research-radar
cd research-radar
pip install requests
```

```bash
ollama pull deepseek-r1:8b   # recommended — better reasoning
ollama pull qwen2.5:7b       # faster, lighter
```

## Usage

```bash
python radar.py                      # last 7 days
python radar.py --days 14            # last 14 days
python radar.py --max 50             # analyze up to 50 papers
python radar.py --model qwen2.5:7b   # different model
python radar.py --no-llm             # keyword filter only, no LLM
```

Reports are saved to `./reports/radar_YYYY-MM-DD.md`.

## How the filter works

Stage 1 asks: does this paper propose a concrete, implementable algorithm that can be directly added to BenderEdge? Strict yes/no. Most papers are eliminated here.

Stage 2 produces, for papers that pass:
- what the paper does
- the specific technique or algorithm
- which file in BenderEdge to modify and how
- expected gain (specific, not generic)
- difficulty: easy / medium / hard
- score 1–5 (most papers land at 2–3; 5 = implement this week)

## Customize for your own project

Edit `BENDER_COMPONENTS` in `radar.py` to describe your own codebase. The LLM will filter and analyze papers against your specific files and modules.

## Sources

| Feed | Coverage |
|------|----------|
| ArXiv q-fin.TR | Trading & Market Microstructure |
| ArXiv q-fin.PM | Portfolio Management |
| ArXiv q-fin.ST | Statistical Finance |
| ArXiv q-fin.RM | Risk Management |
| ArXiv cs.AI | AI papers mentioning finance/trading |
| ArXiv cs.LG | ML papers mentioning finance/trading |

## Part of BenderEdge ecosystem

```
BenderLabs
├── research-radar   — you are here
├── BenderEdge       — multi-agent stock research platform
└── BenderQuant      — XGBoost financial ML engine
```

---

*Built by [Yusuf Bender](https://github.com/yusufbender) · Not financial advice*