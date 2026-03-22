"""
plot_per_asset.py
Per-asset OOS equity breakdown chart for README.
"""
import sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ASSETS = ['ES', 'NQ', 'RTY', 'FDAX', 'FESX', 'CL']
ASSET_HORIZONS = {'ES': 10, 'NQ': 15, 'RTY': 20, 'FDAX': 10, 'FESX': 10, 'CL': 10}
INITIAL_CAPITAL = 100_000
COMM = 0.00006
EQ_WEIGHT = 4.0 / 6    # 0.667x per asset, 4x total
OOS_START = '2024-01-01'

COLORS = {
    'ES':   '#2980B9', 'NQ':   '#E74C3C', 'RTY':  '#27AE60',
    'FDAX': '#8E44AD', 'FESX': '#F39C12', 'CL':   '#D35400',
}
LONG_NAMES = {
    'ES':   'S&P 500 (ES)',
    'NQ':   'Nasdaq 100 (NQ)',
    'RTY':  'Russell 2000 (RTY)',
    'FDAX': 'DAX (FDAX)',
    'FESX': 'EuroStoxx 50 (FESX)',
    'CL':   'Crude Oil (CL)',
}
Q_LONG, Q_SHORT = 0.70, 0.30


def make_pos_pnl(wfo, horizon):
    pos = pd.Series(0.0, index=wfo.index)
    for win_id in wfo['window'].unique():
        m     = wfo['window'] == win_id
        sig   = wfo.loc[m, 'p_ens']
        noise = wfo.loc[m, 'noise'].iloc[0]
        thr_hi = sig.quantile(Q_LONG)
        thr_lo = sig.quantile(Q_SHORT)
        lm = m & (wfo['p_ens'] >= thr_hi)
        sm = m & (wfo['p_ens'] <= thr_lo)
        noisy = (wfo['p_lgb'].abs() < noise) & (wfo['p_cat'].abs() < noise)
        pos[lm & ~noisy] =  1.0
        pos[sm & ~noisy] = -1.0
    tgt = wfo['target'].fillna(0)
    raw = pos * tgt / horizon
    chg = pos.diff().abs().fillna(0) > 0
    pnl = raw - chg.astype(float) * COMM
    return pos, pnl


# ── Load OOS data ─────────────────────────────────────────────────────────────
pnl_oos = {}
for ticker in ASSETS:
    pkl = ('results/ES/wfo_oos_10d.pkl' if ticker == 'ES'
           else f'results/{ticker}/wfo_oos.pkl')
    wfo = pd.read_pickle(pkl)
    _, pnl = make_pos_pnl(wfo, ASSET_HORIZONS[ticker])
    pnl_oos[ticker] = pnl

# Align all to common OOS index
common = sorted(set.intersection(*[set(p.index) for p in pnl_oos.values()]))
common = [i for i in common if str(i) >= OOS_START]
idx    = pd.DatetimeIndex(common)

# ── Per-asset levered equity ───────────────────────────────────────────────────
asset_eq    = {}
asset_stats = {}
for ticker in ASSETS:
    p   = pnl_oos[ticker].reindex(idx).fillna(0)
    lev = EQ_WEIGHT * p
    cum = lev.cumsum()
    eq  = INITIAL_CAPITAL * (1 + cum)
    net = float(eq.iloc[-1] - INITIAL_CAPITAL)
    sh  = float(p.mean() / (p.std() + 1e-8) * np.sqrt(252))
    mdd = float((cum - cum.cummax()).min() * 100)
    pf  = float(p[p > 0].sum() / (abs(p[p < 0].sum()) + 1e-8))
    asset_eq[ticker]    = eq
    asset_stats[ticker] = dict(net=net, sh=sh, mdd=mdd, pf=pf)
    print(f"  {ticker}: Net=${net:+,.0f}  Sh={sh:+.2f}  MDD={mdd:+.1f}%  PF={pf:.2f}")

# Portfolio
port_pnl = sum(pnl_oos[t].reindex(idx).fillna(0) * EQ_WEIGHT for t in ASSETS)
port_cum  = port_pnl.cumsum()
port_eq   = INITIAL_CAPITAL * (1 + port_cum)
port_net  = float(port_eq.iloc[-1] - INITIAL_CAPITAL)
port_sh   = float(port_pnl.mean() / (port_pnl.std() + 1e-8) * np.sqrt(252))
port_mdd  = float((port_cum - port_cum.cummax()).min() * 100)
print(f"\n  PORTFOLIO: Net=${port_net:+,.0f}  Sh={port_sh:+.2f}  MDD={port_mdd:+.1f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 19))
outer = gridspec.GridSpec(3, 1, height_ratios=[2.0, 2.5, 1.0],
                          hspace=0.46, figure=fig)

# ─── Panel 1: Portfolio overview + individual curves ─────────────────────────
ax_port = fig.add_subplot(outer[0])

ax_port.fill_between(port_eq.index, port_eq / 1000, 100,
                     where=port_eq >= INITIAL_CAPITAL,
                     color='#1A252F', alpha=0.10, label='_nolegend_')
ax_port.plot(port_eq.index, port_eq / 1000, color='#1A252F', lw=2.8, zorder=5,
             label=f'Portfolio total  Net ${port_net:+,.0f}  Sh={port_sh:+.2f}  '
                   f'MaxDD={port_mdd:+.1f}%')

for ticker in ASSETS:
    eq = asset_eq[ticker]
    st = asset_stats[ticker]
    ax_port.plot(eq.index, eq / 1000, color=COLORS[ticker], lw=1.5, alpha=0.75,
                 ls='--',
                 label=f"{ticker} h={ASSET_HORIZONS[ticker]}d  "
                       f"${st['net']:+,.0f}  Sh={st['sh']:+.2f}")

ax_port.axhline(100, color='gray', lw=0.9, ls=':', alpha=0.7)
ax_port.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
ax_port.set_ylabel('Account Value', fontsize=10)
ax_port.set_title(
    'OOS 2024–2026 | Portfolio vs Individual Assets  |  $100k Capital  |  '
    '0.667x Leverage per Asset (Equal-Weight, 4x total)\n'
    'Commission: 0.006% RT  |  Regime Exit: ON  |  '
    'Ensemble: LightGBM 50% + CatBoost 50%',
    fontsize=10
)
ax_port.legend(fontsize=8.0, loc='upper left', ncol=2, framealpha=0.92,
               edgecolor='#cccccc')
ax_port.grid(alpha=0.18)

# ─── Panel 2: 6 individual subplots ──────────────────────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                         hspace=0.56, wspace=0.30)
for i, ticker in enumerate(ASSETS):
    ax = fig.add_subplot(inner[i // 3, i % 3])
    eq = asset_eq[ticker]
    st = asset_stats[ticker]
    c  = COLORS[ticker]

    ax.fill_between(eq.index, eq / 1000, 100,
                    where=eq >= INITIAL_CAPITAL, color=c, alpha=0.18)
    ax.fill_between(eq.index, eq / 1000, 100,
                    where=eq < INITIAL_CAPITAL,  color='#C0392B', alpha=0.22)
    ax.plot(eq.index, eq / 1000, color=c, lw=2.0)
    ax.axhline(100, color='gray', lw=0.8, ls=':', alpha=0.6)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
    ax.set_title(f"{LONG_NAMES[ticker]}  |  h={ASSET_HORIZONS[ticker]}d",
                 fontsize=8.5, fontweight='bold', color=c)

    stats_txt = (f"Net:    ${st['net']:+,.0f}\n"
                 f"Sharpe: {st['sh']:+.2f}\n"
                 f"MaxDD:  {st['mdd']:+.1f}%\n"
                 f"PF:     {st['pf']:.2f}")
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=7.8, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=c, alpha=0.90, lw=1.2))
    ax.grid(alpha=0.18)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_fontsize(7)

# ─── Panel 3: Bar chart sorted by P&L ────────────────────────────────────────
ax_bar = fig.add_subplot(outer[2])
tickers_s = sorted(ASSETS, key=lambda t: asset_stats[t]['net'], reverse=True)
nets      = [asset_stats[t]['net'] for t in tickers_s]
sharpes   = [asset_stats[t]['sh']  for t in tickers_s]
mddpcts   = [asset_stats[t]['mdd'] for t in tickers_s]
bar_cols  = [COLORS[t] for t in tickers_s]

bars = ax_bar.bar(tickers_s, nets, color=bar_cols, alpha=0.85,
                  edgecolor='white', lw=1.3, width=0.55)

for bar, net, sh, mdd in zip(bars, nets, sharpes, mddpcts):
    sign = np.sign(net)
    yoff = abs(net) * 0.04 * sign
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                net + yoff,
                f'${net/1000:+.1f}k\nSh={sh:+.2f}\nDD={mdd:+.1f}%',
                ha='center', va='bottom' if net >= 0 else 'top',
                fontsize=8.5, fontweight='bold', color='#1A252F')

ax_bar.axhline(0, color='k', lw=0.9)
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))
ax_bar.set_ylabel('OOS Net P&L ($)', fontsize=9)
ax_bar.set_title(
    'OOS Net P&L per Asset  |  Sorted by Profitability  |  '
    'EW 0.667x Leverage  |  Commission 0.006% RT',
    fontsize=9
)
ax_bar.grid(axis='y', alpha=0.18)
for lbl in ax_bar.get_xticklabels():
    lbl.set_fontsize(10)
    lbl.set_fontweight('bold')

fig.suptitle(
    'Chain of Alpha  —  Multi-Asset Portfolio Performance  |  OOS 2024–2026\n'
    'Monthly WFO  |  LightGBM + CatBoost Ensemble  |  '
    '300 Alpha Factors per Asset (Single-Asset + Intermarket)',
    fontsize=12, fontweight='bold', y=0.999
)

out = 'results/meta/per_asset_oos_breakdown.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved -> {out}')
