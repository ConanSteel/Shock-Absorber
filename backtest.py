# backtest_final.py
# S03 "Shock Absorber" — Final Specification
# Adaptive Lookback (75%) + Carry Continuous (25%), All-TSMOM, Full Harness
# Outputs all metrics, stress tests, regime analysis, and series for paper_figures.py

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# --- Parameters ---
TRAIN_END = '2014-12-31'
TEST_START = '2015-01-01'
SAMPLE_START = '2001-01-01'

VOL_WINDOW = 60
REBAL_FREQ = 'ME'
PM_WEIGHT = 0.15
SECTOR_WEIGHT = (1.0 - PM_WEIGHT) / 3.0
VOL_TARGET = 0.10

# Signal blend — LOCKED
ADAPT_WEIGHT = 0.75
CARRY_WEIGHT = 0.25

# Timing overlay (C continuous)
CONT_FLOOR = 0.4
CONT_CARRY_WT = 0.3
CONT_BREADTH_WT = 0.3

TC_MAP = {
    'CL=F': 1.5, 'BZ=F': 1.5, 'HO=F': 1.5, 'RB=F': 1.5, 'NG=F': 4.0,
    'HG=F': 1.5,
    'GC=F': 1.5, 'SI=F': 1.5, 'PL=F': 3.0, 'PA=F': 3.0,
    'ZC=F': 2.5, 'ZS=F': 2.5, 'ZW=F': 2.5, 'ZL=F': 2.5,
    'KC=F': 7.5, 'SB=F': 7.5, 'CT=F': 7.5, 'CC=F': 7.5,
    'HE=F': 7.5, 'LE=F': 7.5,
}

SECTORS = {
    'Energy':    ['CL=F', 'BZ=F', 'HO=F', 'RB=F', 'NG=F'],
    'IndMetals': ['HG=F'],
    'PreciousMetals': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
    'Agriculture': ['ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'KC=F', 'SB=F', 'CT=F', 'CC=F', 'HE=F', 'LE=F'],
}
SECTOR_ORDER = ['Energy', 'IndMetals', 'PreciousMetals', 'Agriculture']

SECTOR_CARRY_COL = {
    'Energy': 'energy_ts', 'IndMetals': 'indmetals_ts',
    'PreciousMetals': 'pm_ts', 'Agriculture': 'ag_ts',
}

def ann_return(r):
    total = (1 + r).prod()
    ny = len(r) / 12
    if ny <= 0 or total <= 0: return np.nan
    return total ** (1 / ny) - 1

def ann_vol(r): return r.std() * np.sqrt(12)

def sharpe(r):
    v = ann_vol(r)
    if v == 0 or np.isnan(v): return np.nan
    return ann_return(r) / v

def max_drawdown(r):
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def calmar(r):
    dd = max_drawdown(r)
    if dd == 0 or np.isnan(dd): return np.nan
    return ann_return(r) / abs(dd)

def win_rate(r): return (r > 0).mean()

# =====================================================================
# DATA LOADING
# =====================================================================
print("=" * 80)
print("S03 FINAL SPECIFICATION: Adaptive 75% + Carry 25%")
print("=" * 80)
print("\nLoading data...")

daily_ret = pd.read_csv('data/futures_returns.csv', index_col=0, parse_dates=True)
prices_raw = pd.read_csv('data/futures_prices_raw.csv', index_col=0, parse_dates=True)
fundamentals = pd.read_csv('data/fundamentals.csv', index_col=0, parse_dates=True)
benchmarks = pd.read_csv('data/benchmarks.csv', index_col=0, parse_dates=True)

all_tickers = [t for sec in SECTORS.values() for t in sec if t in daily_ret.columns]
monthly_log_ret = daily_ret[all_tickers].resample(REBAL_FREQ).sum()
monthly_ret = np.exp(monthly_log_ret) - 1
monthly_ret = monthly_ret.loc[SAMPLE_START:]
monthly_prices = prices_raw[all_tickers].resample(REBAL_FREQ).last().loc[SAMPLE_START:]
daily_vol_raw = daily_ret[all_tickers].rolling(VOL_WINDOW).std() * np.sqrt(252)
monthly_vol = daily_vol_raw.resample(REBAL_FREQ).last().loc[SAMPLE_START:]
fundamentals = fundamentals.loc[SAMPLE_START:]
benchmarks = benchmarks.loc[SAMPLE_START:]

if 'DTB3_BEY' in benchmarks.columns:
    tbill_monthly = benchmarks['DTB3_BEY'].reindex(monthly_ret.index, method='ffill') / 12 / 100
    tbill_monthly = tbill_monthly.fillna(0)
else:
    tbill_monthly = pd.Series(0.0, index=monthly_ret.index)

inv_vol = 1.0 / monthly_vol.replace(0, np.nan)
tc_df = pd.DataFrame(0.0, index=monthly_ret.index, columns=all_tickers)
for tk in all_tickers:
    if tk in TC_MAP: tc_df[tk] = TC_MAP[tk]
vix_monthly = fundamentals['VIX'].reindex(monthly_ret.index, method='ffill') if 'VIX' in fundamentals.columns else pd.Series(20.0, index=monthly_ret.index)

print(f"  {len(all_tickers)} contracts, {monthly_ret.shape[0]} months")

# =====================================================================
# SIGNAL CONSTRUCTION
# =====================================================================
print("Building signals...")

cum_ret = {}
for lb in [6, 12]:
    cum_ret[lb] = monthly_ret.rolling(lb).apply(lambda x: (1+x).prod()-1, raw=False)

ts_cond = pd.DataFrame(index=monthly_ret.index)
if 'BZ=F' in monthly_prices.columns and 'CL=F' in monthly_prices.columns:
    bz_cl = np.log(monthly_prices['BZ=F'] / monthly_prices['CL=F'])
    ts_cond['energy_ts'] = (bz_cl - bz_cl.rolling(60).mean()) / bz_cl.rolling(60).std()
if 'US_INDPRO' in fundamentals.columns:
    ip = fundamentals['US_INDPRO'].reindex(monthly_ret.index, method='ffill')
    ip_mom = ip.pct_change(6)
    ts_cond['indmetals_ts'] = (ip_mom - ip_mom.rolling(60).mean()) / ip_mom.rolling(60).std()
if 'REAL_YIELD_10Y' in fundamentals.columns:
    ry = fundamentals['REAL_YIELD_10Y'].reindex(monthly_ret.index, method='ffill')
    ry_chg = ry.diff(3)
    ts_cond['pm_ts'] = -((ry_chg - ry_chg.rolling(60).mean()) / ry_chg.rolling(60).std())
if 'USD_INDEX' in fundamentals.columns:
    usd = fundamentals['USD_INDEX'].reindex(monthly_ret.index, method='ffill')
    usd_mom = usd.pct_change(6)
    ts_cond['ag_ts'] = -((usd_mom - usd_mom.rolling(60).mean()) / usd_mom.rolling(60).std())

ts_cols = [c for c in ['energy_ts', 'indmetals_ts', 'pm_ts', 'ag_ts'] if c in ts_cond.columns]
agg_carry_z = ts_cond[ts_cols].mean(axis=1) if ts_cols else pd.Series(0.0, index=monthly_ret.index)
breadth_12 = (cum_ret[12] > 0).sum(axis=1) / cum_ret[12].notna().sum(axis=1)
breadth_12 = breadth_12.fillna(0.5)

# Adaptive lookback
portfolio_vol = daily_ret[all_tickers].rolling(60).std().mean(axis=1) * np.sqrt(252)
portfolio_vol_m = portfolio_vol.resample(REBAL_FREQ).last().loc[SAMPLE_START:]
vol_pctile = portfolio_vol_m.expanding(60).rank(pct=True)

adapt_sig = pd.DataFrame(0.0, index=monthly_ret.index, columns=all_tickers)
for t in range(len(monthly_ret.index)):
    dt = monthly_ret.index[t]
    vp = vol_pctile.loc[dt] if dt in vol_pctile.index else 0.5
    if pd.isna(vp): vp = 0.5
    w_short = np.clip(vp, 0.2, 0.8)
    w_long = 1.0 - w_short
    sig_6 = (cum_ret[6].loc[dt] > 0).astype(float).reindex(all_tickers).fillna(0) if dt in cum_ret[6].index else pd.Series(0.0, index=all_tickers)
    sig_12 = (cum_ret[12].loc[dt] > 0).astype(float).reindex(all_tickers).fillna(0) if dt in cum_ret[12].index else pd.Series(0.0, index=all_tickers)
    adapt_sig.loc[dt] = w_short * sig_6 + w_long * sig_12

carry_sig = pd.DataFrame(0.0, index=monthly_ret.index, columns=all_tickers)
for sector in SECTOR_ORDER:
    tickers = [t for t in SECTORS[sector] if t in monthly_ret.columns]
    col = SECTOR_CARRY_COL.get(sector)
    if col and col in ts_cond.columns:
        z = ts_cond[col].reindex(monthly_ret.index).fillna(0)
        for tk in tickers:
            carry_sig[tk] = z.clip(-2, 2) / 2

combined_sig = adapt_sig * ADAPT_WEIGHT + carry_sig * CARRY_WEIGHT
tsmom_sig = (cum_ret[12] > 0).astype(float)

# =====================================================================
# HARNESS
# =====================================================================
def build_weights(signal_df):
    wsw = {}
    for sector in SECTOR_ORDER:
        tickers = [t for t in SECTORS[sector] if t in signal_df.columns]
        if not tickers:
            wsw[sector] = pd.DataFrame(0.0, index=monthly_ret.index, columns=tickers)
            continue
        sig = signal_df[tickers].copy()
        iv = inv_vol[tickers].copy()
        iv_norm = iv.div(iv.sum(axis=1), axis=0).fillna(0)
        wsw[sector] = sig * iv_norm
    return wsw

def run_harness(sector_weights):
    pw = pd.DataFrame(0.0, index=monthly_ret.index, columns=all_tickers)
    for t in range(1, len(monthly_ret.index)):
        date = monthly_ret.index[t]
        prev = monthly_ret.index[t - 1]
        for sector in SECTOR_ORDER:
            tickers = [tk for tk in SECTORS[sector] if tk in monthly_ret.columns]
            if not tickers: continue
            sw = PM_WEIGHT if sector == 'PreciousMetals' else SECTOR_WEIGHT
            w_sec = sector_weights[sector]
            if prev in w_sec.index:
                pw.loc[date, tickers] = w_sec.loc[prev, tickers].values * sw
    leverage = pd.Series(1.0, index=monthly_ret.index)
    for t in range(1, len(monthly_ret.index)):
        prev = monthly_ret.index[t - 1]
        cz = agg_carry_z.loc[prev] if prev in agg_carry_z.index else 0.0
        b = breadth_12.loc[prev] if prev in breadth_12.index else 0.5
        if pd.isna(cz): cz = 0.0
        if pd.isna(b): b = 0.5
        carry_clipped = np.clip(cz, -1.0, 1.0)
        breadth_scaled = np.clip((b - 0.3) / 0.4, 0.0, 1.0)
        lev = CONT_FLOOR + CONT_CARRY_WT * carry_clipped + CONT_BREADTH_WT * breadth_scaled
        leverage.iloc[t] = np.clip(lev, 0.3, 1.0)
    pw_lev = pw.mul(leverage, axis=0)
    ret_unsc = (pw_lev.shift(1) * monthly_ret).sum(axis=1)
    tvol = ret_unsc.rolling(3).std() * np.sqrt(12)
    vscalar = (VOL_TARGET / tvol).clip(upper=1.0).fillna(1.0)
    pw_sc = pw_lev.mul(vscalar, axis=0)
    gross = (pw_sc.shift(1) * monthly_ret).sum(axis=1)
    gross_coll = gross + tbill_monthly.reindex(monthly_ret.index).fillna(0)
    turnover = pw_sc.diff().abs()
    tc_cost = (turnover * tc_df / 10000).sum(axis=1)
    net = gross_coll - tc_cost
    return net, pw_sc, leverage

# =====================================================================
# RUN
# =====================================================================
print("Running full model (Adapt 75% + Carry 25%)...")
model_w = build_weights(combined_sig)
net_ret, pw_scaled, leverage_ts = run_harness(model_w)

print("Running baseline (all-TSMOM, in harness)...")
base_w = build_weights(tsmom_sig)
baseline_net, _, _ = run_harness(base_w)

# =====================================================================
# HEADLINE PERFORMANCE
# =====================================================================
print("\n" + "=" * 80)
print("HEADLINE PERFORMANCE — FINAL SPECIFICATION")
print("Adaptive Lookback 75% + Carry Continuous 25%, All-TSMOM, Full Harness")
print("=" * 80)

r = net_ret.dropna()
train_r = r[r.index <= TRAIN_END]
test_r = r[r.index >= TEST_START]

for tag, sub in [('Full Sample', r), ('Train (2001-2014)', train_r), ('Test (2015-2025)', test_r)]:
    ar = ann_return(sub) * 100
    av = ann_vol(sub) * 100
    sr = sharpe(sub)
    md = max_drawdown(sub) * 100
    cr = calmar(sub)
    wr = win_rate(sub) * 100
    print(f"\n  {tag}:")
    print(f"    Ann. Return:  {ar:+.2f}%")
    print(f"    Volatility:   {av:.2f}%")
    print(f"    Sharpe:       {sr:.3f}")
    print(f"    Max Drawdown: {md:.1f}%")
    print(f"    Calmar:       {cr:.3f}")
    print(f"    Win Rate:     {wr:.1f}%")
    print(f"    N months:     {len(sub)}")

avg_lev = leverage_ts.loc[r.index].mean()
avg_lev_train = leverage_ts.loc[train_r.index].mean()
avg_lev_test = leverage_ts.loc[test_r.index].mean()
print(f"\n  Avg Leverage: Full={avg_lev:.3f}, Train={avg_lev_train:.3f}, Test={avg_lev_test:.3f}")
tbill_avg = tbill_monthly.mean() * 12
print(f"  T-bill avg annual: {tbill_avg*100:.2f}%")
gross_exp = pw_scaled.abs().sum(axis=1)
print(f"  Avg Gross Exposure: {gross_exp.mean():.1%}")
tc_annual = (pw_scaled.diff().abs() * tc_df / 10000).sum(axis=1).mean() * 12
print(f"  TC drag (annual): {tc_annual*100:.2f}%")

# =====================================================================
# BASELINE COMPARISON
# =====================================================================
print("\n" + "=" * 80)
print("BASELINE COMPARISON (ALL-TSMOM IN HARNESS)")
print("=" * 80)
bl = baseline_net.dropna()
for tag, sub in [('Full', bl), ('Train', bl[bl.index <= TRAIN_END]), ('Test', bl[bl.index >= TEST_START])]:
    print(f"  {tag}: Sharpe={sharpe(sub):.3f}, Return={ann_return(sub)*100:.2f}%, "
          f"Vol={ann_vol(sub)*100:.2f}%, MaxDD={max_drawdown(sub)*100:.1f}%")

# =====================================================================
# STRESS TESTS
# =====================================================================
print("\n" + "=" * 80)
print("STRESS TESTS")
print("=" * 80)
stress_periods = {
    'COVID (Feb-Apr 2020)': ('2020-02-01', '2020-04-30'),
    'GFC H2 (Jul-Dec 2008)': ('2008-07-01', '2008-12-31'),
    'Lost Decade (2011-2020)': ('2011-01-01', '2020-12-31'),
    'Taper Tantrum (May-Aug 2013)': ('2013-05-01', '2013-08-31'),
}
stress_results = []
for name, (start, end) in stress_periods.items():
    mask = (r.index >= start) & (r.index <= end)
    sr = r[mask]
    sl = leverage_ts.loc[r.index][mask]
    if len(sr) < 2: continue
    cum = (1 + sr).cumprod()
    ptd = ((cum - cum.cummax()) / cum.cummax()).min()
    tr = (1 + sr).prod() - 1
    res = {'Period': name, 'Total Return': tr * 100, 'Max DD': ptd * 100,
           'Avg Leverage': sl.mean(), 'Min Leverage': sl.min(), 'N months': len(sr)}
    if len(sr) >= 12:
        res['Ann. Return'] = ann_return(sr) * 100
        res['Sharpe'] = sharpe(sr)
    stress_results.append(res)
    print(f"\n  {name}:")
    print(f"    Total return: {res['Total Return']:+.2f}%, Max DD: {res['Max DD']:.1f}%")
    print(f"    Avg leverage: {res['Avg Leverage']:.3f} (min: {res['Min Leverage']:.3f})")
    if 'Sharpe' in res: print(f"    Sharpe: {res['Sharpe']:.3f}")

for period, (s, e) in [('COVID', ('2020-02-01', '2020-04-30')), ('Normal 2017-19', ('2017-01-01', '2019-12-31'))]:
    mask = (monthly_ret.index >= s) & (monthly_ret.index <= e)
    rets = monthly_ret.loc[mask, all_tickers]
    if len(rets) > 2:
        corr = rets.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        print(f"\n  {period} avg cross-contract corr: {upper.stack().mean():.3f}")

# =====================================================================
# REGIME-CONDITIONAL ANALYSIS
# =====================================================================
print("\n" + "=" * 80)
print("REGIME-CONDITIONAL ANALYSIS")
print("=" * 80)
vix_at_ret = vix_monthly.reindex(r.index).ffill()
regimes = {'VIX < 20': vix_at_ret < 20, 'VIX 20-30': (vix_at_ret >= 20) & (vix_at_ret <= 30), 'VIX > 30': vix_at_ret > 30}
regime_results = []
for rname, mask in regimes.items():
    rr = r[mask]
    rl = leverage_ts.loc[r.index][mask]
    if len(rr) < 6: continue
    rcorr = monthly_ret.loc[rr.index, all_tickers].corr()
    upper_rc = rcorr.where(np.triu(np.ones(rcorr.shape), k=1).astype(bool))
    res = {'Regime': rname, 'Ann. Return': ann_return(rr)*100, 'Volatility': ann_vol(rr)*100,
           'Sharpe': sharpe(rr), 'Max DD': max_drawdown(rr)*100, 'Avg Leverage': rl.mean(),
           'Avg Cross-Corr': upper_rc.stack().mean(), 'N months': len(rr), 'Pct': len(rr)/len(r)*100}
    regime_results.append(res)
    print(f"\n  {rname} ({len(rr)}m, {res['Pct']:.0f}%):")
    print(f"    Return={res['Ann. Return']:+.2f}%, Vol={res['Volatility']:.2f}%, "
          f"Sharpe={res['Sharpe']:.3f}, DD={res['Max DD']:.1f}%")
    print(f"    Leverage={res['Avg Leverage']:.3f}, X-Corr={res['Avg Cross-Corr']:.3f}")

# =====================================================================
# ROLLING 10-YEAR OOS
# =====================================================================
print("\n" + "=" * 80)
print("ROLLING 10-YEAR OOS")
print("=" * 80)
rolling_results = []
for sy in range(2001, 2017):
    ey = sy + 9
    if ey > max(r.index.year): break
    rw = r[(r.index.year >= sy) & (r.index.year <= ey)].dropna()
    if len(rw) < 12: continue
    m = {'Start': sy, 'End': ey, 'Sharpe': sharpe(rw), 'Ann. Return': ann_return(rw)*100, 'Max DD': max_drawdown(rw)*100}
    rolling_results.append(m)
    tag = " ***" if m['Sharpe'] < 0 else ""
    print(f"  {sy}-{ey}: Sharpe={m['Sharpe']:.3f}, Ret={m['Ann. Return']:.2f}%, DD={m['Max DD']:.1f}%{tag}")

test_ret = r[r.index >= TEST_START].dropna()
n_roll, n_neg = 0, 0
for i in range(len(test_ret) - 11):
    w = test_ret.iloc[i:i+12]
    if len(w) == 12:
        n_roll += 1
        if ann_return(w) < 0: n_neg += 1
print(f"\n  12m test windows: {n_neg}/{n_roll} negative ({n_neg/n_roll:.1%})")

# =====================================================================
# BENCHMARK COMPARISON
# =====================================================================
print("\n" + "=" * 80)
print("BENCHMARKS")
print("=" * 80)
bench_results = [{'Name': 'S03 Full Model', 'Ann. Return': ann_return(r)*100, 'Volatility': ann_vol(r)*100,
                  'Sharpe': sharpe(r), 'Max DD': max_drawdown(r)*100, 'Corr w/ S03': 1.0}]
tbill_cum = tbill_monthly.reindex(r.index).fillna(0)
bench_results.append({'Name': 'T-bill', 'Ann. Return': ann_return(tbill_cum)*100, 'Volatility': ann_vol(tbill_cum)*100,
                      'Sharpe': np.nan, 'Max DD': max_drawdown(tbill_cum)*100, 'Corr w/ S03': r.corr(tbill_cum)})
gsci_cols = [c for c in benchmarks.columns if 'GSCI' in c.upper() or 'GSG' in c.upper()]
if gsci_cols:
    gr = benchmarks[gsci_cols[0]].reindex(r.index).fillna(0)
    bench_results.append({'Name': 'GSCI TR Proxy', 'Ann. Return': ann_return(gr)*100, 'Volatility': ann_vol(gr)*100,
                          'Sharpe': sharpe(gr), 'Max DD': max_drawdown(gr)*100, 'Corr w/ S03': r.corr(gr)})
aqr_cols = [c for c in benchmarks.columns if 'AQR' in c.upper() or 'VME' in c.upper() or 'CMTY' in c.upper()]
for ac in aqr_cols[:2]:
    ar = benchmarks[ac].reindex(r.index).fillna(0)
    cm = r.dropna().index.intersection(ar.dropna().index)
    if len(cm) > 24:
        bench_results.append({'Name': f'AQR ({ac})', 'Ann. Return': ann_return(ar.loc[cm])*100,
                              'Volatility': ann_vol(ar.loc[cm])*100, 'Sharpe': sharpe(ar.loc[cm]),
                              'Max DD': max_drawdown(ar.loc[cm])*100, 'Corr w/ S03': r.loc[cm].corr(ar.loc[cm])})

print(f"\n  {'Name':<25s} | {'Ret%':>6s} | {'Vol%':>5s} | {'Sharpe':>6s} | {'DD%':>5s} | {'Corr':>5s}")
print("  " + "-" * 65)
for b in bench_results:
    s = f"{b['Sharpe']:.3f}" if not pd.isna(b['Sharpe']) else "  N/A"
    print(f"  {b['Name']:<25s} | {b['Ann. Return']:>6.2f} | {b['Volatility']:>5.2f} | {s:>6s} | {b['Max DD']:>5.1f} | {b['Corr w/ S03']:>+5.2f}")

# =====================================================================
# TC SENSITIVITY
# =====================================================================
print("\n" + "=" * 80)
print("TC SENSITIVITY")
print("=" * 80)
for mult, label in [(1, '1x base'), (2, '2x base'), (3, '3x base')]:
    tc_t = {k: v*mult for k, v in TC_MAP.items()}
    td = pd.DataFrame(0.0, index=monthly_ret.index, columns=all_tickers)
    for tk in all_tickers:
        if tk in tc_t: td[tk] = tc_t[tk]
    g = (pw_scaled.shift(1) * monthly_ret).sum(axis=1)
    gc = g + tbill_monthly.reindex(monthly_ret.index).fillna(0)
    tc = (pw_scaled.diff().abs() * td / 10000).sum(axis=1)
    nr = (gc - tc).dropna()
    print(f"  {label}: Sharpe={sharpe(nr):.3f}, Return={ann_return(nr)*100:.2f}%")

g = (pw_scaled.shift(1) * monthly_ret).sum(axis=1)
gc = g + tbill_monthly.reindex(monthly_ret.index).fillna(0)
to = pw_scaled.diff().abs()
tc_reg = pd.Series(0.0, index=monthly_ret.index)
for idx in to.index:
    v = vix_monthly.loc[idx] if idx in vix_monthly.index else 20.0
    m = 2.0 if (not pd.isna(v) and v > 30) else 1.0
    rc = 0.0
    for tk in all_tickers:
        if tk in TC_MAP: rc += to.at[idx, tk] * TC_MAP[tk] * m / 10000
    tc_reg.at[idx] = rc
nr_reg = (gc - tc_reg).dropna()
print(f"  Regime (2x VIX>30): Sharpe={sharpe(nr_reg):.3f}, Return={ann_return(nr_reg)*100:.2f}%")

# =====================================================================
# SAVE
# =====================================================================
print("\nSaving outputs...")
net_ret.to_csv('results/final_net_returns.csv', header=['net_return'])
baseline_net.to_csv('results/baseline_b_returns.csv', header=['net_return'])
leverage_ts.to_csv('results/final_leverage.csv', header=['leverage'])
pd.DataFrame(stress_results).to_csv('results/stress_tests.csv', index=False)
pd.DataFrame(regime_results).to_csv('results/regime_analysis.csv', index=False)
pd.DataFrame(bench_results).to_csv('results/final_benchmarks.csv', index=False)
pd.DataFrame(rolling_results).to_csv('results/rolling_oos.csv', index=False)
print("All outputs saved. Run paper_figures.py next.")
print("=" * 80)
