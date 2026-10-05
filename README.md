# Shock Absorber: A Systematic Long-Only Commodity Futures Strategy

Adaptive time-series momentum (75%) blended with sector-specific carry signals (25%), applied to 20 commodity futures contracts across four sectors. Monthly rebalancing with a carry-momentum timing overlay and 10% annualised volatility target.

**Full-sample (Jan 2001 -- Dec 2025):** Sharpe 1.04, annualised return 5.78%, max drawdown -9.8%.

## Structure

```
backtest.py          # Main backtest script (run first)
paper_figures.py     # Publication figures F1-F7 (run after backtest.py)
requirements.txt
data/                # Input CSVs (not included -- see below)
results/             # Backtest output CSVs
figures/             # Generated figures
latex/               # Paper source (main_submission.tex)
```

## Data

The backtest expects four CSV files in `data/`:

| File | Contents | Source |
|---|---|---|
| `futures_returns.csv` | Daily log returns, 20 contracts | Yahoo Finance (yfinance) |
| `futures_prices_raw.csv` | Daily close prices | Yahoo Finance |
| `fundamentals.csv` | VIX, US IP, real yields, USD index | FRED |
| `benchmarks.csv` | T-bill rate, GSCI proxy, AQR factors | FRED / AQR |

All data was sourced from publicly available APIs. Raw data is not committed to avoid redistribution issues.

## Usage

```bash
pip install -r requirements.txt
python backtest.py
python paper_figures.py
```

`backtest.py` prints full performance metrics to stdout and saves seven CSV files to `results/`. `paper_figures.py` reads those CSVs and generates the seven figures referenced in the paper.

## Strategy Summary

- **Signals:** Adaptive TSMOM (6m/12m lookback blend weighted by cross-sectional volatility percentile) + sector-specific carry z-scores (Brent-WTI spread, IP momentum, real yield changes, USD momentum).
- **Universe:** 20 commodity futures -- Energy (5), Industrial Metals (1), Precious Metals (4), Agriculture (10).
- **Risk management:** Inverse-volatility position sizing, carry-momentum timing overlay (leverage range 0.3-1.0), one-sided 10% vol target (scale down only).
- **Transaction costs:** Contract-specific bps (1.5-7.5 bps). Annual TC drag ~0.05%. Strategy is effectively TC-immune.

## Author

Conan Trindade Steel
