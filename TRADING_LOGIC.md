# Trading Logic (v3)

This is not a live-money trading bot. It is an automated participant for the
[Explaining Markets](https://explainingmarkets.ai/) forecasting competition:
it receives a webhook when a company reports earnings, predicts a single
percentile score, and submits that score to the competition API. No orders
are placed and no capital is at risk — only the competition's own scoring
is affected.

v3 extends the v2 model (see the other repo, `EarningsReleaseTradingBot`)
with industry/peer-relative features and a market-volatility feature.

## What it predicts

`predicted_percentile` — a float in `[0, 1]` representing where the ticker's
next-day abnormal (market-adjusted) return (`CAR1`) is expected to rank
against every other event scored that quarter. `0` = the quarter's worst
reaction, `0.5` = median, `1` = the best. It is a **cross-sectional rank**,
not a percentile within the ticker's own history.

## End-to-end flow

1. Competition platform POSTs a signed webhook to the Modal endpoint when an
   earnings event fires.
2. `modal_app.py` verifies the HMAC signature against the raw request body
   using `EM_WEBHOOK_SECRET`, then ACKs within 20 seconds.
3. A background job runs `predict.py::predict(event)` (delegates to
   `model_v3/v3_predictor.py::V3Predictor`), with a 5-minute budget from the
   ACK.
4. It fetches the event's summary JSON (`information_url`) for the
   earnings-call facts and earnings-surprise metric.
5. `FeatureExtractor` builds 10 features, `V3Predictor` runs them through a
   pre-trained linear regression, converts the CAR1 forecast to a
   percentile with a sigmoid, and submits it via `EM_API_KEY`.

## Model

`sklearn.linear_model.LinearRegression`, 10 features:

```
predicted_car1 = features · coefficients + intercept
predicted_percentile = sigmoid(predicted_car1 * sigmoid_scale)   # sigmoid_scale = 3
```

Coefficients, intercept, and sigmoid scale are frozen at
`model_artifacts/model_params.json`, produced by `train_model_v3.py`. Not
retrained automatically — a static snapshot from whenever that script was
last run.

## Features (10)

| # | Feature | Coefficient | Source | Lookback |
|---|---------|:-----------:|--------|----------|
| 1 | `baseline_gemini` | +0.162 | Gemini (`gemini-flash-latest`), prompted with the earnings-call facts and a fixed up/neutral/down calibration (~25/50/25% base rates). 2 retries, 30s timeout. Falls back to `0.5` if `GEMINI_API_KEY` is unset or the call fails. | none |
| 2 | `sentiment_positive` | +0.099 | FinBERT (`ProsusAI/finbert`, run locally) softmax probability the earnings-call facts read "positive" | none |
| 3 | `sentiment_negative` | +0.109 | FinBERT "negative" probability (the "neutral" class is dropped — pos+neg+neu=1 makes it redundant/collinear) | none |
| 4 | `delta_positive` | −0.016 | Change in `sentiment_positive` vs. the ticker's prior quarter | 1 quarter |
| 5 | `delta_negative` | −0.044 | Change in `sentiment_negative` vs. prior quarter | 1 quarter |
| 6 | `car1_lag1` | +0.780 | Ticker's own actual `CAR1` from the previous quarter (momentum) | 1 quarter |
| 7 | `surprise` | −0.024 | Reported earnings surprise from the event's `earnings_surprise` metric; `0.0` if unavailable | none |
| 8 | `car1_vs_peers_lag1` | −0.814 | `car1_lag1` minus the ticker's **industry**-average CAR1 from the previous quarter | 1 quarter |
| 9 | `surprise_vs_peers_lag1` | +0.024 | `surprise` minus the ticker's industry-average surprise from the previous quarter | 1 quarter |
| 10 | `r_m_volatility` | −0.323 | 60-trading-day rolling std. dev. of SPY (S&P 500 ETF) daily returns | 60 trading days |

Note the two largest-magnitude coefficients by far are `car1_lag1` (+0.78)
and `car1_vs_peers_lag1` (−0.81) — the model leans heavily on last quarter's
own return relative to its industry peers, more than on any single-quarter
sentiment or surprise signal.

## Peer/industry tracking (`model_v3/peer_tracker.py`)

- 2,601 tickers are mapped to one of 16 industry buckets
  (`model_artifacts/ticker_industry_map.json`); an unrecognized ticker is
  auto-classified as `"Other"` and uses the cross-industry average.
- Industry-quarter averages start from a historical snapshot
  (`model_artifacts/historical_industry_stats.json`, covering
  `2025_Q4`–`2026_Q4`+) and are updated in-process as new predictions come
  in during the run.
- **This in-process state is not persisted anywhere** (unlike the
  webhook-dedupe store, which uses a `modal.Dict`). A container
  restart/redeploy resets peer stats back to the historical snapshot,
  discarding anything learned from predictions made since the last cold
  start.

## Market volatility (`model_v3/alpha_vantage_client.py`)

- Pulls SPY daily prices from Alpha Vantage (`TIME_SERIES_DAILY`,
  `outputsize=full`, ~20 years of history per fetch) and computes a 60-day
  rolling standard deviation of daily returns.
- Cached in memory for 1 hour before refetching.
- If the event date isn't an exact match in the cached series (e.g. a
  weekend), looks back up to 5 calendar days for the nearest trading day.
- On any fetch failure, or if no data is found at all, falls back to a flat
  placeholder (`r_m = 0.0`, `r_m_volatility = 0.01`) rather than erroring.
- Alpha Vantage free-tier keys are rate-limited (typically 25 requests/day);
  the 1-hour cache keeps normal usage well under that, but a burst of many
  earnings events on the same day could matter if the key isn't a paid tier.

## Training (offline, `train_model_v3.py` — not run at inference time)

- Data: `backtesting/data/EARNINGS_RELEASE_*.jsonl` for quarters
  `2025_Q4`–`2026_Q3`; the saved `model_artifacts/train_stats.json` records
  the actual trained snapshot as 4,315 events spanning
  `2025_Q4`–`2027_Q1` (6 quarters) across 16 industries.
- Reported train R² ≈ **0.138** — the model explains roughly 14% of the
  variance in the scoring metric on its own training data (not a held-out
  test split, unlike the v2 report). `predict.py`'s docstring claims
  "+45% vs. baseline (Gemini + Surprise)" — an internal comparison, not an
  independently audited number. The large majority of return variance
  remains unexplained.

## Failure modes / things to know before relying on this

- Fetch/parse failure, no facts found, or any exception in feature
  extraction or prediction → **silently falls back to a neutral 0.5
  prediction**, still submitted (only the first submission per event is
  scored, so a bad fallback can't be corrected after the fact).
- No `GEMINI_API_KEY` → `baseline_gemini` silently defaults to `0.5`.
- No `ALPHA_VANTAGE_API_KEY`, or the Alpha Vantage call fails →
  `r_m_volatility` silently defaults to `0.01` with no error raised.
- New/unseen ticker → treated as industry `"Other"`; own-history features
  (`delta_positive`, `delta_negative`, `car1_lag1`) fall back to `0.0`.
- Duplicate webhook deliveries (platform retries on 5xx/timeout) are
  deduped via a `modal.Dict` keyed on `Webhook-Id`.
- The repo also contains legacy/auxiliary scripts alongside the live
  `predict.py`: `predict_v2_backup.py`, `train_model.py` (the older v2
  trainer), `verify_bot_backtest_match.py`, and
  `verify_bot_matches_backtest.py`. Only `predict.py` (via `V3Predictor`)
  is what actually runs in production.

## Credentials required (`.env`, gitignored — never commit)

- `EM_API_KEY` — competition submission auth
- `EM_WEBHOOK_SECRET` — verifies incoming webhook signatures
- `GEMINI_API_KEY` — optional; without it `baseline_gemini` silently uses
  the `0.5` fallback described above
- `ALPHA_VANTAGE_API_KEY` — optional; without it `r_m_volatility` silently
  uses the `0.01` fallback described above
