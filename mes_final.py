"""
mes_final.py -- MES Financial Wrapper over monthly_wfo PnL logic
=================================================================
PnL mechanism: IDENTICAL to monthly_wfo.py (pos * target_10d / FORWARD_DAYS).
Financial layer:
  - Start: $100,000
  - Position: fixed MES contracts (1:1 leverage, recalc at period start)
  - Commission: $1.50 per round-turn per contract (charged on trade close)
  - Output: equity in dollars, trade stats, one chart.
"""

import json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from monthly_wfo import (
    build_feature_matrix, select_stable_factors,
    rolling_wfo as run_monthly_wfo,
    make_positions, backtest,
    POOL_FILE, TOP_N_FACTORS, FORWARD_DAYS,
    Q_LONG, Q_SHORT,
    WFO_START, WFO_END, OOS_START, OOS_END,
)
from generate_intermarket import load_all_aligned, log_ret

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INITIAL_CAPITAL = 100_000
MES_MULT        = 5        # $5 per ES point
COMM_RT         = 1.50     # $ per contract per round-turn

STRATEGIES = {
    'LightGBM': ('p_lgb', False),
    'CatBoost': ('p_cat', False),
    'Ensemble': ('p_ens', False),
}
COLORS = {
    'LightGBM': '#2980B9',
    'CatBoost': '#E74C3C',
    'Ensemble': '#27AE60',
    'Buy & Hold': '#95A5A6',
}
eps = 1e-8


# ─────────────────────────────────────────────
# FINANCIAL WRAPPER
# ─────────────────────────────────────────────
def dollar_wrap(pos: pd.Series,
                target_10d: pd.Series,
                es_close: pd.Series,
                initial_capital: float) -> dict:
    """
    Converts monthly_wfo % PnL to dollar equity.

    PnL logic (UNCHANGED from monthly_wfo):
        pnl_pct[t] = pos[t] * target_10d[t] / FORWARD_DAYS

    Financial layer:
        - n_contracts = floor(initial_capital / (MES_MULT * es_price_at_start))
        - notional    = n_contracts * MES_MULT * es_price_at_start
        - dollar_gross[t] = pnl_pct[t] * notional
        - commission charged = COMM_RT * n_contracts on each trade close
    """
    es = es_close.reindex(pos.index).ffill()
    tgt = target_10d.reindex(pos.index).fillna(0)

    # Fixed contracts for the period (1:1 leverage at start)
    es_entry = float(es.iloc[0])
    n_cont = max(1, int(initial_capital / (MES_MULT * es_entry)))
    notional = n_cont * MES_MULT * es_entry

    # ── Same PnL formula as monthly_wfo ──────────────────────────────
    pnl_pct    = pos * tgt / FORWARD_DAYS
    dollar_gross = pnl_pct * notional   # scaled by notional, not capital

    # ── Commission: $1.50 * n_cont charged when a trade CLOSES ───────
    # A trade closes when: pos_prev != 0  AND  pos_curr != pos_prev
    pos_prev    = pos.shift(1).fillna(0)
    close_mask  = (pos_prev != 0) & (pos != pos_prev)   # exit or reversal
    comm_series = pd.Series(0.0, index=pos.index)
    comm_series[close_mask] = COMM_RT * n_cont

    # Close any position still open at the last bar
    if pos.iloc[-1] != 0:
        comm_series.iloc[-1] += COMM_RT * n_cont

    dollar_net = dollar_gross - comm_series
    equity     = initial_capital + dollar_net.cumsum()

    # ── Trade-level stats ─────────────────────────────────────────────
    dir_chg = (pos != pos.shift(1)).cumsum()
    active  = pos != 0

    if active.any():
        grp          = dir_chg[active]
        trade_net    = dollar_net[active].groupby(grp).sum()
        gross_wins   = trade_net.clip(lower=0).sum()
        gross_losses = (-trade_net).clip(lower=0).sum()
        n_trades     = len(trade_net)
        pf           = gross_wins / (gross_losses + eps)
        avg_trade    = trade_net.mean()
    else:
        n_trades = gross_wins = gross_losses = avg_trade = 0
        pf = 0.0

    peak   = equity.cummax()
    max_dd = (equity - peak).min()

    return dict(
        equity       = equity,
        pnl_pct      = pnl_pct,
        dollar_net   = dollar_net,
        comm         = comm_series,
        net_profit   = equity.iloc[-1] - initial_capital,
        max_dd       = max_dd,
        total_comm   = comm_series.sum(),
        n_trades     = n_trades,
        profit_factor= pf,
        avg_trade    = avg_trade,
        gross_wins   = gross_wins,
        gross_losses = gross_losses,
        n_cont       = n_cont,
        notional     = notional,
        cum_ret_pct  = (equity.iloc[-1] / initial_capital - 1) * 100,
    )


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
def print_table(results: dict):
    w = 95
    print('\n' + '=' * w)
    print(f"{'MES TRADING REPORT':^{w}}")
    print(f"{'$100,000 capital | $1.50 RT/contract | pos * target_10d / FORWARD_DAYS (unchanged)':^{w}}")
    print('=' * w)
    hdr = (f"{'Strategy':<12} {'Period':<11} {'N_cont':>6} {'Final($)':>10}"
           f" {'Net($)':>9} {'Ret%':>7} {'MaxDD($)':>10}"
           f" {'Trades':>7} {'Comm($)':>8} {'AvgTrd($)':>10} {'PF':>5}")
    print(hdr)
    print('-' * w)

    for name, strat_res in results.items():
        for period, fb, is_oos in strat_res:
            rf_tag = ' [OOS]' if is_oos else ' [VAL]'
            print(f"{name:<12} {period:<11} {fb['n_cont']:>6}"
                  f" {fb['equity'].iloc[-1]:>10,.0f}"
                  f" {fb['net_profit']:>+9,.0f}"
                  f" {fb['cum_ret_pct']:>+6.1f}%"
                  f" {fb['max_dd']:>10,.0f}"
                  f" {fb['n_trades']:>7}"
                  f" {fb['total_comm']:>8,.0f}"
                  f" {fb['avg_trade']:>+10,.0f}"
                  f" {fb['profit_factor']:>5.2f}")
        print('-' * w)

    print('PF = Profit Factor (Gross Wins / Gross Losses).  MaxDD in dollars from equity peak.')
    print('=' * w)


def print_detail(name: str, strat_res: list):
    print(f"\n  -- {name} ------------------------------------------")
    for period, fb, _ in strat_res:
        print(f"  {period}:")
        print(f"    Contracts:         {fb['n_cont']} MES  "
              f"(notional ${fb['notional']:,.0f})")
        print(f"    Final balance:     ${fb['equity'].iloc[-1]:>12,.0f}")
        print(f"    Net profit:        ${fb['net_profit']:>+12,.0f}  "
              f"({fb['cum_ret_pct']:+.2f}%)")
        print(f"    Max drawdown:      ${fb['max_dd']:>12,.0f}")
        print(f"    Total commissions: ${fb['total_comm']:>12,.0f}")
        print(f"    Trade count:       {fb['n_trades']:>12}")
        print(f"    Avg net/trade:     ${fb['avg_trade']:>+12,.0f}")
        print(f"    Profit factor:     {fb['profit_factor']:>12.3f}"
              f"  (wins ${fb['gross_wins']:,.0f} / "
              f"losses ${fb['gross_losses']:,.0f})")
        print()


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_equity(results: dict, close: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3, 1.2]})
    fig.suptitle(
        'MES Equity Curve  |  $100,000 start  |  $1.50 RT/contract\n'
        f'Monthly WFO  |  Q{int(Q_LONG*100)}/Q{int(Q_SHORT*100)} signals  |  '
        f'PnL = pos × target_10d / {FORWARD_DAYS}  (same as monthly_wfo.py)',
        fontsize=11, y=0.99)

    oos_ts = pd.Timestamp(OOS_START)
    ax = axes[0]

    # ── Buy & Hold MES ───────────────────────────────────────────────
    bh_price  = close.loc[WFO_START:OOS_END]
    bh_entry  = float(bh_price.iloc[0])
    bh_n      = max(1, int(INITIAL_CAPITAL / (MES_MULT * bh_entry)))
    bh_notional = bh_n * MES_MULT * bh_entry
    # Use log return * notional for smooth B&H (same spirit as strategy P&L)
    bh_ret = bh_price.pct_change().fillna(0)
    bh_eq  = INITIAL_CAPITAL + (bh_ret * bh_notional).cumsum()

    ax.plot(bh_eq.index, bh_eq.values / 1_000, color=COLORS['Buy & Hold'],
            lw=1.5, ls='--', alpha=0.7,
            label=f"Buy & Hold MES ({bh_n} contracts)  "
                  f"${bh_eq.iloc[-1]:,.0f}")

    # ── Strategy lines ───────────────────────────────────────────────
    for name, strat_res in results.items():
        c    = COLORS[name]
        # Stitch VAL and OOS equity into one series
        eq_v = strat_res[0][1]['equity']   # VAL
        eq_o = strat_res[1][1]['equity']   # OOS
        eq_all = pd.concat([eq_v, eq_o])

        final = eq_o.iloc[-1]
        lw    = 2.4 if name == 'Ensemble' else 1.8
        ax.plot(eq_all.index, eq_all.values / 1_000, color=c, lw=lw,
                label=f"{name}  ${final:,.0f}  "
                      f"(net {eq_o.iloc[-1]-INITIAL_CAPITAL:+,.0f})")

    ax.axhline(INITIAL_CAPITAL / 1_000, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(oos_ts, color='red', lw=1.6, ls=':', label='OOS start 2024-01-01')
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], oos_ts, pd.Timestamp(OOS_END),
                     color='red', alpha=0.05)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Equity ($000)', fontsize=11)
    ax.set_title('Equity Curve  —  VAL 2021–2024  |  OOS 2024–2026', fontsize=10)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))

    # ── Underwater drawdown (Ensemble) ───────────────────────────────
    ax2   = axes[1]
    eq_v  = results['Ensemble'][0][1]['equity']
    eq_o  = results['Ensemble'][1][1]['equity']
    eq_all = pd.concat([eq_v, eq_o])
    uw_all = eq_all - eq_all.cummax()

    ax2.fill_between(uw_all.index, uw_all.values / 1_000, 0,
                     where=uw_all.values < 0,
                     color='#E74C3C', alpha=0.45, label='Ensemble drawdown')
    ax2.plot(uw_all.index, uw_all.values / 1_000, color='#C0392B', lw=1.0, alpha=0.8)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.axvline(oos_ts, color='red', lw=1.5, ls=':')
    ymin2 = min(uw_all.min() / 1_000 * 1.15, -0.5)
    ax2.fill_betweenx([ymin2, 0], oos_ts, pd.Timestamp(OOS_END),
                      color='red', alpha=0.05)
    ax2.set_ylim(ymin2, 0.5)
    ax2.set_ylabel('Drawdown ($000)', fontsize=10)
    ax2.set_title('Underwater Drawdown from Equity Peak  —  Ensemble', fontsize=10)
    ax2.legend(fontsize=8.5, loc='lower left')
    ax2.grid(alpha=0.25)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('mes_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved -> mes_final.png')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 68)
    print('MES FINANCIAL WRAPPER  |  same PnL logic as monthly_wfo.py')
    print('=' * 68)

    # ── Load ──────────────────────────────────────────────────────────
    print('\nStep 1 -- Loading...')
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool = json.load(f)
    pool_names = [r['name'] for r in pool]

    full   = load_all_aligned()
    feat_df = build_feature_matrix(full, pool_names)
    close  = full['es_close']
    ret_1d = log_ret(close)
    target = np.log(close.shift(-FORWARD_DAYS) / close)
    print(f'  ES price at WFO_START ({WFO_START}): '
          f'{close.loc[WFO_START:].iloc[0]:.0f}')
    print(f'  ES price at OOS_START ({OOS_START}): '
          f'{close.loc[OOS_START:].iloc[0]:.0f}')

    # ── Factor selection ──────────────────────────────────────────────
    print('\nStep 2 -- Factor selection...')
    top_factors = select_stable_factors(pool, feat_df, target)

    # ── Monthly WFO ───────────────────────────────────────────────────
    print(f'\nStep 3 -- Monthly WFO VAL ({WFO_START} -> {WFO_END})...')
    wfo_val = run_monthly_wfo(feat_df, target, top_factors, WFO_START, WFO_END)

    print(f'\nStep 4 -- Monthly WFO OOS ({OOS_START} -> {OOS_END})...')
    wfo_oos = run_monthly_wfo(feat_df, target, top_factors, OOS_START, OOS_END)

    # ── Verify % returns match monthly_wfo exactly ────────────────────
    print('\n  [Sanity check — % returns must match monthly_wfo.py]')
    for name, (col, re) in STRATEGIES.items():
        rv = backtest(wfo_val, ret_1d, col, re)
        ro = backtest(wfo_oos, ret_1d, col, re)
        print(f'  {name:<12}  VAL={rv["cum_ret"]:+.2f}%  OOS={ro["cum_ret"]:+.2f}%')

    # ── Dollar wrapper ────────────────────────────────────────────────
    print('\nStep 5 -- Dollar wrapper...')
    results = {}  # name -> [(period_label, fb, is_oos), ...]

    for name, (col, re) in STRATEGIES.items():
        pos_val = make_positions(wfo_val, col, re)
        pos_oos = make_positions(wfo_oos, col, re)

        fb_v = dollar_wrap(pos_val, wfo_val['target'], close,
                           initial_capital=INITIAL_CAPITAL)
        # OOS starts from where VAL ended
        fb_o = dollar_wrap(pos_oos, wfo_oos['target'], close,
                           initial_capital=fb_v['equity'].iloc[-1])

        results[name] = [
            ('VAL 2021-2024', fb_v, False),
            ('OOS 2024-2026', fb_o, True),
        ]

    # ── Print ─────────────────────────────────────────────────────────
    print_table(results)

    print('\nDetailed breakdown:')
    for name, strat_res in results.items():
        print_detail(name, strat_res)

    # Buy & Hold reference
    bh_entry = float(close.loc[WFO_START:].iloc[0])
    bh_n     = max(1, int(INITIAL_CAPITAL / (MES_MULT * bh_entry)))
    bh_not   = bh_n * MES_MULT * bh_entry
    bh_ret   = close.loc[WFO_START:OOS_END].pct_change().fillna(0)
    bh_eq    = INITIAL_CAPITAL + (bh_ret * bh_not).cumsum()
    print(f'  -- Buy & Hold MES ({bh_n} contracts, notional ${bh_not:,.0f}) --')
    print(f'    VAL end:   ${float(bh_eq.loc[:WFO_END].iloc[-1]):,.0f}')
    print(f'    OOS end:   ${float(bh_eq.iloc[-1]):,.0f}')
    print(f'    Net total: +${float(bh_eq.iloc[-1]) - INITIAL_CAPITAL:,.0f}')

    # ── Chart ─────────────────────────────────────────────────────────
    print('\nStep 6 -- Chart...')
    plot_equity(results, close)

    # ── Summary ───────────────────────────────────────────────────────
    ens_v = results['Ensemble'][0][1]
    ens_o = results['Ensemble'][1][1]
    print('\n' + '=' * 68)
    print('BOTTOM LINE — Ensemble:')
    print(f'  Start:         ${INITIAL_CAPITAL:,}')
    print(f'  VAL end:       ${ens_v["equity"].iloc[-1]:,.0f}'
          f'  ({ens_v["cum_ret_pct"]:+.1f}%)')
    print(f'  OOS end:       ${ens_o["equity"].iloc[-1]:,.0f}'
          f'  ({ens_o["cum_ret_pct"]:+.1f}%)')
    print(f'  OOS max DD:    ${ens_o["max_dd"]:,.0f}')
    print(f'  Total comm:    ${ens_v["total_comm"]+ens_o["total_comm"]:,.0f}'
          f'  (negligible vs strategy edge)')
    print('=' * 68)

    print(f'\nTotal elapsed: {time.time()-t0:.0f}s')
    print('Saved: mes_final.png')
