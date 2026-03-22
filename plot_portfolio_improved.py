"""
plot_portfolio_improved.py
Перестраивает portfolio_equity.png с нормальным масштабом:
  - Верхний панель: каждый актив отдельно, ось Y от ~$85k до ~$200k
  - Нижний панель: портфель (сумма P&L, нормированный к 0%)
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR    = 'results'
INITIAL_CAPITAL = 100_000
OOS_START      = '2024-01-01'

TARGET_ASSETS = ['ES', 'NQ', 'RTY', 'FDAX', 'FESX', 'E6', 'CL']

COLORS = {
    'ES':   '#2980B9',
    'NQ':   '#E74C3C',
    'RTY':  '#27AE60',
    'FDAX': '#8E44AD',
    'FESX': '#F39C12',
    'E6':   '#1ABC9C',
    'CL':   '#D35400',
}

# ── Загрузка сохранённых результатов ──────────────────────────────────────────
def load_asset(ticker):
    out_dir = os.path.join(RESULTS_DIR, ticker)
    wfo_val = pd.read_pickle(os.path.join(out_dir, 'wfo_val.pkl'))
    wfo_oos = pd.read_pickle(os.path.join(out_dir, 'wfo_oos.pkl'))
    with open(os.path.join(out_dir, 'summary.json')) as f:
        summary = json.load(f)
    return wfo_val, wfo_oos, summary


def rebuild_equity(wfo, horizon, q_long=0.70, q_short=0.30,
                   comm_rate=0.0003, capital=INITIAL_CAPITAL):
    """Rebuild equity curve from saved WFO DataFrame."""
    # Positions per window
    pos = pd.Series(0.0, index=wfo.index)
    noise_col = 'noise'
    for win_id in wfo['window'].unique():
        m      = wfo['window'] == win_id
        sig    = wfo.loc[m, 'p_ens']
        noise  = wfo.loc[m, noise_col].iloc[0]
        thr_hi = sig.quantile(q_long)
        thr_lo = sig.quantile(q_short)
        long_m  = m & (wfo['p_ens'] >= thr_hi)
        short_m = m & (wfo['p_ens'] <= thr_lo)
        # Regime exit
        noisy   = (wfo['p_lgb'].abs() < noise) & (wfo['p_cat'].abs() < noise)
        long_m  = long_m  & ~noisy
        short_m = short_m & ~noisy
        pos[long_m]  =  1.0
        pos[short_m] = -1.0

    tgt_fwd   = wfo['target'].fillna(0)
    pnl_raw   = pos * tgt_fwd / horizon
    trade_chg = pos.diff().abs().fillna(0) > 0
    pnl_net   = pnl_raw - trade_chg.astype(float) * comm_rate
    equity    = capital * (1 + pnl_net.cumsum())
    return equity, pos


# ── Построение улучшенного графика ────────────────────────────────────────────
def plot_improved():
    asset_data = {}
    for ticker in TARGET_ASSETS:
        try:
            wfo_val, wfo_oos, summary = load_asset(ticker)
            horizon = summary['horizon']
            eq_val, pos_val = rebuild_equity(wfo_val, horizon)
            eq_oos, pos_oos = rebuild_equity(wfo_oos, horizon)
            asset_data[ticker] = {
                'eq_val': eq_val,
                'eq_oos': eq_oos,
                'horizon': horizon,
                'val_net': summary['val']['net_profit'],
                'oos_net': summary['oos']['net_profit'],
                'val_sharpe': summary['val']['sharpe'],
                'oos_sharpe': summary['oos']['sharpe'],
                'oos_maxdd':  summary['oos']['max_dd_pct'],
            }
            print(f'  Loaded {ticker}')
        except Exception as e:
            print(f'  [WARN] {ticker}: {e}')

    if not asset_data:
        print('No data loaded')
        return

    # ── Figure: 3 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs  = fig.add_gridspec(3, 1, height_ratios=[2.8, 1.6, 1.0], hspace=0.38)
    ax1 = fig.add_subplot(gs[0])   # individual equity curves ($85k-$200k)
    ax2 = fig.add_subplot(gs[1])   # portfolio cumulative P&L (%)
    ax3 = fig.add_subplot(gs[2])   # OOS bar chart

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1: Per-asset equity, normalised scale $100k base
    # ─────────────────────────────────────────────────────────────────────────
    port_val_pnl = None
    port_oos_pnl = None

    for ticker, d in asset_data.items():
        c      = COLORS.get(ticker, '#555555')
        eq_v   = d['eq_val']
        eq_o   = d['eq_oos']
        # Offset OOS so it continues from end of VAL visually
        val_end_equity = eq_v.iloc[-1]
        eq_o_shifted   = eq_o + (val_end_equity - INITIAL_CAPITAL)

        lbl = (f"{ticker}  h={d['horizon']}d  "
               f"OOS {d['oos_net']:+,.0f}$  Sh={d['oos_sharpe']:+.2f}")

        ax1.plot(eq_v.index,      eq_v.values / 1000,            color=c, lw=1.8, label=lbl)
        ax1.plot(eq_o_shifted.index, eq_o_shifted.values / 1000, color=c, lw=1.8, ls='--', alpha=0.75)

        # Accumulate portfolio
        pnl_v = eq_v - INITIAL_CAPITAL
        pnl_o = eq_o - INITIAL_CAPITAL
        port_val_pnl = pnl_v if port_val_pnl is None else port_val_pnl.add(pnl_v, fill_value=0)
        port_oos_pnl = pnl_o if port_oos_pnl is None else port_oos_pnl.add(pnl_o, fill_value=0)

    ax1.axvline(pd.Timestamp(OOS_START), color='red', lw=1.5, ls=':', label='OOS start')
    ax1.axhline(INITIAL_CAPITAL / 1000, color='k', lw=0.8, ls=':')
    ax1.set_ylim(82, 210)           # <-- нормальный масштаб
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
    ax1.set_ylabel('Account Value ($)', fontsize=10)
    ax1.set_title(
        'Per-Asset Equity Curves  |  Solid = VAL 2021-2024  |  Dashed = OOS 2024-2026\n'
        'Each account starts at $100k (1:1 leverage, monthly WFO Ensemble+Regime)',
        fontsize=10
    )
    ax1.legend(fontsize=7.5, loc='upper left', ncol=2,
               framealpha=0.85, edgecolor='#cccccc')
    ax1.grid(alpha=0.2)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2: Portfolio cumulative P&L (%)
    # ─────────────────────────────────────────────────────────────────────────
    n_assets    = len(asset_data)
    total_cap   = n_assets * INITIAL_CAPITAL

    # Normalise to % of total capital
    port_val_pct = (port_val_pnl / total_cap * 100)
    port_oos_off = port_val_pct.iloc[-1]
    port_oos_pct = (port_oos_pnl / total_cap * 100) + port_oos_off

    ax2.fill_between(port_val_pct.index, port_val_pct.values, 0,
                     where=port_val_pct.values >= 0,
                     color='#27AE60', alpha=0.25, label='VAL profit')
    ax2.fill_between(port_val_pct.index, port_val_pct.values, 0,
                     where=port_val_pct.values < 0,
                     color='#E74C3C', alpha=0.25)
    ax2.plot(port_val_pct.index, port_val_pct.values,
             color='#27AE60', lw=2.2, label=f'Portfolio VAL  ({port_val_pct.iloc[-1]:+.1f}%)')
    ax2.plot(port_oos_pct.index, port_oos_pct.values,
             color='#27AE60', lw=2.2, ls='--', alpha=0.85,
             label=f'Portfolio OOS  ({port_oos_pnl.iloc[-1]/total_cap*100:+.1f}%)')

    ax2.axvline(pd.Timestamp(OOS_START), color='red', lw=1.5, ls=':')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.1f}%'))
    ax2.set_ylabel('Portfolio Return (%)', fontsize=10)
    ax2.set_title(
        f'Portfolio Cumulative Return  |  $700k total ({n_assets} x $100k)\n'
        f'VAL total P&L: ${port_val_pnl.iloc[-1]:+,.0f}   |   '
        f'OOS total P&L: ${port_oos_pnl.iloc[-1]:+,.0f}',
        fontsize=10
    )
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.2)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3: OOS per-asset bar (sorted)
    # ─────────────────────────────────────────────────────────────────────────
    tickers_sorted = sorted(asset_data.keys(),
                            key=lambda t: asset_data[t]['oos_net'], reverse=True)
    oos_nets    = [asset_data[t]['oos_net'] for t in tickers_sorted]
    oos_sharpes = [asset_data[t]['oos_sharpe'] for t in tickers_sorted]
    bar_cols    = [COLORS.get(t, '#555') for t in tickers_sorted]

    bars = ax3.bar(tickers_sorted, oos_nets, color=bar_cols, alpha=0.82,
                   edgecolor='white', lw=1.2)
    for bar, v, sh in zip(bars, oos_nets, oos_sharpes):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 v + 300 * np.sign(v),
                 f'${v/1000:+.1f}k\nSh={sh:+.2f}',
                 ha='center', va='bottom' if v >= 0 else 'top',
                 fontsize=8, fontweight='bold')
    ax3.axhline(0, color='k', lw=0.8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax3.set_ylabel('OOS Net P&L', fontsize=9)
    ax3.set_title('OOS 2024-2026 Net P&L per Asset (sorted, with Sharpe)', fontsize=9)
    ax3.grid(axis='y', alpha=0.2)

    fig.suptitle(
        'Multi-Asset Alpha Portfolio  —  Monthly WFO LightGBM+CatBoost Ensemble+Regime\n'
        '300 alpha factors per asset (SA + Intermarket)  |  Top-50 stable factors  |  $100k per asset',
        fontsize=11, y=0.995, fontweight='bold'
    )

    save_path = os.path.join(RESULTS_DIR, 'portfolio', 'portfolio_equity_v2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {save_path}')


if __name__ == '__main__':
    print('Rebuilding improved portfolio chart...')
    plot_improved()
    print('Done.')
