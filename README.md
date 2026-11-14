# Football Predictions Agent

An accurate, testable football predictions agent for Python 3.12. It combines
**Elo ratings** (K=20, home advantage 60 Elo) with a **Dixon–Coles Poisson
goal model**, blended with recent form (last 5), head-to-head, rest days, and a
logistic calibration layer. Every prediction is validated by walk-forward
backtesting — no look-ahead.

> **No guarantee of profit. This is a statistical model, not betting advice.**
> Probabilities are estimates; football is noisy and bookmaker odds include a
> margin. Never bet money you cannot afford to lose.

## Quickstart (no API key needed)

```bash
pip install -r requirements.txt
python -m pytest -q
python -m football_agent.cli predict --home Arsenal --away Chelsea
python -m football_agent.cli backtest --league PL
```

Or install the entry point: `pip install -e .` then use `football-predict`.

## Commands

| Command | What it does |
|---|---|
| `football-predict predict --home Arsenal --away Chelsea` | 1X2 probs, xG, top-5 scorelines, BTTS, over/under 2.5, confidence + top-3 factors |
| `football-predict fixtures --league PL --date 2026-11-15` | Predict that day's fixtures (live API) |
| `football-predict backtest --league PL` | Walk-forward backtest on history |
| `football-predict evaluate --league PL` | Backtest + plain-language verdict |
| `football-predict fetch --league PL` | Fetch live data (needs API key) |
| `football-predict agent --home Arsenal Liverpool --away Chelsea "Man City"` | Rank fixtures, write `output/predictions.csv` + `output/predictions.md` |

## Connect to ChatGPT

Give ChatGPT your team names and get a prediction **from this model**,
not from ChatGPT guessing: deploy the FastAPI wrapper, create a Custom
GPT, and add it as an Action. Full non-technical steps:
[`chatgpt/README.md`](chatgpt/README.md).

```bash
pip install -r requirements.txt
uvicorn football_agent.api:app --reload   # or: python -m football_agent.api
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

Endpoints: `GET /health`, `POST /predict`, `POST /predict-batch`,
`GET /fixtures?league=PL&date=2026-11-15`, `GET /backtest?league=PL`.
Team names are fuzzy-matched ("Man Utd", "Spurs" work); an unknown name
returns 400 with "did you mean" suggestions. The fitted model is cached
per league. If `API_KEY` is set, every endpoint except `/health` requires
an `X-API-Key` header. Deploy free on Render with the included
`render.yaml`, or with the included `Dockerfile`.

## Web chat page (free, no ChatGPT plan needed)

If your ChatGPT account can't create Custom GPTs, use the web chat page
served by the same API — type two team names and get the model's
prediction straight away:

- Live: **https://football-predictions-agent-agk8.onrender.com/**
- Locally: `uvicorn football_agent.api:app` then open
  http://localhost:8000/

The page is a chat: it shows a "waking the model up…" state, calls
`GET /health` first to wake the free host, then `POST /predict`, and
renders the reply as a chat bubble with win/draw/away % bars, expected
goals, most likely outcome + confidence, top-5 scorelines, BTTS and
over/under 2.5, and the model's factors. Unknown team names show the
"did you mean" suggestions as clickable chips; timeouts offer a Retry
button. On Render's free plan the first request after idle can take
50–90s while the service wakes; once warm, predictions take a few
seconds.

## How accuracy is measured

`backtest` uses **walk-forward, expanding-window** evaluation: for each match
after a 100-match warm-up, the model is fit only on matches played before it,
predicts it, then moves on. Reported metrics:

- **Accuracy (1X2)** — fraction of matches where the most likely outcome won.
  A naive "always home" baseline scores ~47% on the sample data, so accuracy
  alone is a weak signal.
- **Brier score** — mean squared error of the probability vector (0 = perfect;
  0.667 = always predicting 1/3 each). The honest headline metric.
- **Log loss** — penalises confident wrong predictions heavily.
- **ROI vs odds** — flat 1-unit stakes on the most-likely outcome using the
  odds in the data (only when an odds CSV is supplied / present in sample data).

Run it yourself: `python -m football_agent.cli backtest` prints the real table
on the bundled sample data.

## Data

- **Primary:** [football-data.org](https://www.football-data.org/) v4, free tier
  (10 requests/minute, competitions PL, PD, SA, BL1, FL1, CL).
  Get a free key at https://www.football-data.org/client/register, then:
  ```bash
  cp .env.example .env   # then set FOOTBALL_DATA_API_KEY
  export FOOTBALL_DATA_API_KEY=your_key_here
  football-predict fetch --league PL
  ```
  API responses are cached for 6h in `data/cache/`.
- **Fallback (offline):** bundled sample dataset
  `data/sample/pl_matches.csv` — 480 realistic Premier League matches
  (2024-25 full season, 380 + 2025-26 first 10 rounds, 100), generated with a
  seeded round-robin schedule and team strengths so home-win rate (~47%),
  goals (~1.59 home / 1.27 away) and scorelines look like real PL data.
  Any CSV with columns `date,season,league,home_team,away_team,home_goals,
  away_goals,home_odds,draw_odds,away_odds` works via `CsvProvider`.

## Model

1. **Elo** — ratings start at 1500, K=20, home advantage +60 Elo.
2. **Dixon–Coles Poisson** — per-team attack/defence strengths (log scale,
   mean-centred), fitted by maximum likelihood; rho correction for low scores
   (0-0, 1-0, 0-1, 1-1). Outputs a full 11×11 scoreline matrix.
3. **Adjustments** — expected goals are nudged by form difference (last 5),
   head-to-head points, and rest days (<=3 days vs >3).
4. **Calibration** — multinomial logistic regression over raw probs + Elo +
   form difference, fit on out-of-sample predictions; falls back to raw
   probabilities when data is scarce.

Output per match: P(home/draw/away), xG both teams, top-5 scorelines, P(BTTS),
P(over 2.5), confidence (max outcome prob), and the top 3 driving factors.

## Limitations

- Sample data is synthetic (seeded, realistic but not real results); live
  accuracy should be measured on real data via the free API.
- The model uses only match results — no line-ups, injuries, xG from
  Opta-style event data, or market odds as a feature.
- Low-scoring football means even a good model is ~50-55% accurate on 1X2;
  the edge, if any, is small and variance is large.
- No betting advice is offered or implied.

## Project layout

```
src/football_agent/
  data/      providers (football-data.org, CSV, sample) + Match type
  model/     elo.py, poisson.py (Dixon-Coles), features.py, calibration.py, predict.py
  backtest.py  walk-forward evaluation
  agent.py     rank fixtures -> predictions.csv / predictions.md
  cli.py       football-predict entry point
  api.py       FastAPI wrapper for ChatGPT (see "Connect to ChatGPT")
  static/index.html  free web chat page served at GET / (no ChatGPT needed)
chatgpt/     openapi.yaml, gpt-instructions.md, README.md (Custom GPT setup)
render.yaml  free Render deploy for the API
Dockerfile   container deploy for the API
data/sample/pl_matches.csv   bundled offline dataset (480 matches)
tests/       pytest: Elo, Poisson, Dixon-Coles, no-leakage backtest, CLI smoke, API, web page
```

## License

MIT — see `LICENSE`.
