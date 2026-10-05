# paper_figures.py
# S03 "Shock Absorber" — Publication Figures (F1–F7)
# Updated for final specification: Adaptive 75% + Carry 25%
# Run AFTER backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('figures', exist_ok=True)

TRAIN_END = '2014-12-31'
TEST_START = '2015-01-01'

C_MODEL = '#1f4e79'
C_BASELINE = '#c55a11'
C_GSCI = '#7f7f7f'
C_TBILL = '#548235'
C_FULL = '#1f4e79'
C_TEST = '#c55a11'
C_POS = '#548235'
C_NEG = '#c00000'
DPI = 150


# --- Load ---
print("Loading results...")
net_ret = pd.read_csv('results/final_net_returns.csv', index_col=0, parse_dates=True).squeeze()
baseline_ret = pd.read_csv('results/baseline_b_returns.csv', index_col=0, parse_dates=True).squeeze()
leverage = pd.read_csv('results/final_leverage.csv', index_col=0, parse_dates=True).squeeze()
benchmarks = pd.read_csv('data/benchmarks.csv', index_col=0, parse_dates=True)
rolling_oos = pd.read_csv('results/rolling_oos.csv')

if 'DTB3_BEY' in benchmarks.columns:
    tbill = benchmarks['DTB3_BEY'].reindex(net_ret.index, method='ffill') / 12 / 100
    tbill = tbill.fillna(0)
else:
    tbill = pd.Series(0.0, index=net_ret.index)

gsci_cols = [c for c in benchmarks.columns if 'GSCI' in c.upper() or 'GSG' in c.upper()]
gsci_ret = benchmarks[gsci_cols[0]].reindex(net_ret.index).fillna(0) if gsci_cols else None


# --- F1: Cumulative returns ---
print("F1: Cumulative returns...")
fig, ax = plt.subplots(figsize=(10, 5.5))
r = net_ret.dropna()
ax.plot((1+r).cumprod().index, (1+r).cumprod().values, color=C_MODEL, linewidth=2, label='S03 Full Model')
bl = baseline_ret.reindex(r.index).fillna(0)
ax.plot((1+bl).cumprod().index, (1+bl).cumprod().values, color=C_BASELINE, linewidth=1.5, linestyle='--', label='TSMOM Baseline')
tb = tbill.reindex(r.index).fillna(0)
ax.plot((1+tb).cumprod().index, (1+tb).cumprod().values, color=C_TBILL, linewidth=1.2, linestyle=':', label='T-bill')
if gsci_ret is not None:
    gr = gsci_ret.reindex(r.index).fillna(0)
    ax.plot((1+gr).cumprod().index, (1+gr).cumprod().values, color=C_GSCI, linewidth=1.2, linestyle='-.', label='GSCI TR Proxy')
ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', alpha=0.3, linewidth=0.8, label='Train/Test')
ax.axvspan(pd.Timestamp(TEST_START), r.index[-1], alpha=0.04, color='blue')
ax.set_yscale('log')
ax.set_ylabel('Cumulative Return (log scale)', fontsize=11)
ax.set_title('Figure 1: Cumulative Net Returns, January 2001 – December 2025', fontsize=12)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
fig.tight_layout()
fig.savefig('figures/F1_cumulative_returns.png', dpi=DPI, bbox_inches='tight')
plt.close()


# F2: Rolling 10-year Sharpe
print("F2: Rolling 10-year Sharpe...")
fig, ax = plt.subplots(figsize=(10, 5.5))
labels = [f"{int(row['Start'])}–{int(row['End'])}" for _, row in rolling_oos.iterrows()]
sharpes = rolling_oos['Sharpe'].values
colors = [C_POS if s > 0 else C_NEG for s in sharpes]
x = np.arange(len(labels))
ax.bar(x, sharpes, color=colors, alpha=0.85, width=0.7, edgecolor='white', linewidth=0.5)
fs_sharpe = r.mean() / r.std() * np.sqrt(12)
ax.axhline(fs_sharpe, color=C_MODEL, linestyle='--', alpha=0.6, linewidth=1.2,
           label=f'Full-sample Sharpe ({fs_sharpe:.2f})')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.set_title('Figure 2: Rolling 10-Year Out-of-Sample Sharpe Ratio', fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures/F2_rolling_oos_sharpe.png', dpi=DPI, bbox_inches='tight')
plt.close()


# --- F3: Blend ratio sensitivity ---
# Hardcoded from combo_test.py results
print("F3: Blend ratio sensitivity...")
fig, ax = plt.subplots(figsize=(8, 5))

carry_pcts = [0, 15, 20, 25, 30, 35, 40, 50]
s_full =     [0.859, 0.997, 1.017, 1.039, 1.060, 1.082, 1.105, 1.149]
s_test =     [1.177, 1.148, 1.162, 1.176, 1.188, 1.197, 1.204, 1.208]
rets =       [6.28,  6.11,  5.95,  5.78,  5.60,  5.42,  5.23,  4.83]

ax2 = ax.twinx()
ax.plot(carry_pcts, s_full, 'o-', color=C_FULL, linewidth=2, markersize=7, label='Sharpe (Full)')
ax.plot(carry_pcts, s_test, 's--', color=C_TEST, linewidth=2, markersize=7, label='Sharpe (Test)')
ax2.bar(carry_pcts, rets, width=3, alpha=0.15, color=C_FULL, label='Ann. Return (%)')
ax.axvline(25, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Final spec (25%)')
ax.axhline(1.0, color='grey', linestyle=':', alpha=0.4)
ax2.axhline(5.0, color='#7f7f7f', linestyle='--', alpha=0.5, label='Return floor (5%)')

ax.set_xlabel('Carry Weight (%)', fontsize=11)
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax2.set_ylabel('Annualised Return (%)', fontsize=11, color='grey')
ax.set_title('Figure 3: Carry Blend Ratio Sensitivity', fontsize=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig('figures/F3_blend_sensitivity.png', dpi=DPI, bbox_inches='tight')
plt.close()


# F4: Candidate comparison
# Hardcoded from candidate_tests.py
print("F4: Candidate comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

candidates = ['TSMOM\nBaseline', 'Carry\nBinary', 'Carry-Cont\n70/30', 'Conviction\nSigmoid',
              'Adaptive\nLookback', 'Short\nSymmetric', 'Short\nGated', 'FINAL\nAdapt+Carry']
sharpes_c = [0.859, 0.730, 0.977, 0.901, 0.946, 0.630, 0.772, 1.039]
colors_c = ['#1f4e79', '#c55a11', '#c55a11', '#7030a0', '#548235', '#c00000', '#c00000', '#2e75b6']

x = np.arange(len(candidates))
bars = ax.bar(x, sharpes_c, color=colors_c, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axhline(0.859, color=C_MODEL, linestyle='--', alpha=0.4, label='TSMOM Baseline')
ax.axhline(1.0, color='grey', linestyle=':', alpha=0.3)

ax.set_xticks(x)
ax.set_xticklabels(candidates, fontsize=8)
ax.set_ylabel('Sharpe Ratio (Full Sample)', fontsize=11)
ax.set_title('Figure 4: Enhancement Candidate Comparison', fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures/F4_candidate_comparison.png', dpi=DPI, bbox_inches='tight')
plt.close()


# --- F5: Leverage ---
print("F5: Leverage time series...")
fig, ax = plt.subplots(figsize=(10, 4.5))
lev = leverage.dropna()
ax.fill_between(lev.index, lev.values, alpha=0.3, color=C_MODEL)
ax.plot(lev.index, lev.values, color=C_MODEL, linewidth=0.8)
ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(lev.mean(), color=C_MODEL, linestyle=':', alpha=0.5, label=f'Mean ({lev.mean():.2f})')
ax.set_ylabel('Leverage Scalar', fontsize=11)
ax.set_title('Figure 5: Carry-Momentum Timing Overlay Leverage', fontsize=12)
ax.set_ylim(0.2, 1.05)
ax.legend(fontsize=9, loc='upper right')
ax.annotate('GFC', xy=(pd.Timestamp('2008-10-01'), 0.35), fontsize=8, color='grey')
ax.annotate('COVID', xy=(pd.Timestamp('2020-03-01'), 0.35), fontsize=8, color='grey')
fig.tight_layout()
fig.savefig('figures/F5_leverage_timeseries.png', dpi=DPI, bbox_inches='tight')
plt.close()


# F6: Drawdown comparison
print("F6: Drawdown comparison...")
fig, ax = plt.subplots(figsize=(10, 4.5))
def dd_series(r):
    cum = (1 + r).cumprod()
    return (cum - cum.cummax()) / cum.cummax()
dd_m = dd_series(net_ret.dropna())
dd_b = dd_series(baseline_ret.reindex(net_ret.dropna().index).fillna(0))
ax.fill_between(dd_m.index, dd_m.values * 100, alpha=0.2, color=C_MODEL)
ax.plot(dd_m.index, dd_m.values * 100, color=C_MODEL, linewidth=0.8, label='S03 Full Model')
ax.plot(dd_b.index, dd_b.values * 100, color=C_BASELINE, linewidth=1, linestyle='--', alpha=0.7, label='TSMOM Baseline')
ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.set_title('Figure 6: Underwater Equity Curve', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
fig.tight_layout()
fig.savefig('figures/F6_drawdown.png', dpi=DPI, bbox_inches='tight')
plt.close()


# --- F7: TC sensitivity ---
# Hardcoded from backtest output
print("F7: TC sensitivity...")
fig, ax = plt.subplots(figsize=(7, 5))
tc_labels = ['1x base', '2x base', '3x base', 'Regime\n(2x VIX>30)']
tc_sharpes = [1.039, 1.031, 1.022, 1.037]
tc_colors = [C_POS, C_TEST, C_NEG, '#7030a0']
x = np.arange(len(tc_labels))
ax.bar(x, tc_sharpes, color=tc_colors, alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tc_labels, fontsize=9)
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.set_title('Figure 7: Transaction Cost Sensitivity', fontsize=12)
ax.set_ylim(0.95, 1.08)
ax.axhline(1.0, color='grey', linestyle=':', alpha=0.4)
fig.tight_layout()
fig.savefig('figures/F7_tc_sensitivity.png', dpi=DPI, bbox_inches='tight')
plt.close()


print("\nAll figures saved to figures/")
