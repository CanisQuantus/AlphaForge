"""
risk_audit.py -- Financial Audit & Risk Management
===================================================
Builds on top of monthly_wfo.py:
  1. Volatility Targeting: 10% annual, dynamically scaled by VX level
  2. Transaction costs: 0.03% open / 0.10% close or reversal
  3. Capital tracking (compounding) from $100,000
  4. Equity Curve in dollars + Underwater Drawdown chart
  5. Trade Log Summary (avg profit/trade, total commissions, profit factor)
  6. Recovery Factor for OOS period
"""

import json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# -- Reuse all monthly_wfo machinery -----------------------------------------
from monthly_wfo import (
    build_feature_matrix, select_stable_factors,
    rolling_wfo as run_monthly_wfo,
    make_positions,
    POOL_FILE, TOP_N_FACTORS, FORWARD_DAYS,
    Q_LONG, Q_SHORT, NOISE_PCTILE,
    WFO_START, WFO_END, OOS_START, OOS_END,
)
from generate_intermarket import load_all_aligned, log_ret

# ─────────────────────────────────────────────
# RISK CONFIG
# ─────────────────────────────────────────────
INITIAL_CAPITAL = 100_000     # USD

# Volatility Targeting
VOL_TARGET      = 0.10        # 10% annual
VOL_LOOKBACK    = 20          # trailing days for realized vol
MAX_VOL_SCALAR  = 3.0         # never exceed 3x leverage
MIN_REALVOL     = 0.05        # floor realized vol to avoid crazy scalars

# VX overlay: if VX > VX_NEUTRAL, scale position down proportionally
VX_NEUTRAL      = 18.0        # VX below this -> scalar = 1.0
VX_FLOOR_SCALAR = 0.25        # never scale below 25% of base size

# Transaction costs (as fraction of notional)
COST_OPEN       = 0.0003      # 0.03% on entry
COST_CLOSE      = 0.0010      # 0.10% on exit or reversal

STRATEGIES = {
    'LightGBM':        ('p_lgb', False),
    'CatBoost':        ('p_cat', False),
    'Ensemble':        ('p_ens', False),
    'Ensemble+Regime': ('p_ens', True),
}
COLORS = {
    'LightGBM':        '#2980B9',
    'CatBoost':        '#E74C3C',
    'Ensemble':        '#27AE60',
    'Ensemble+Regime': '#8E44AD',
}
eps = 1e-8


# ─────────────────────────────────────────────
# 1. POSITION SIZING
# ─────────────────────────────────────────────
def compute_size_scalar(ret_1d: pd.Series, vx_daily: pd.Series,
                        index: pd.DatetimeIndex) -> pd.Series:
    """
    Returns daily size multiplier (float >= 0) for each date in index.
    Multiplier = vol_scalar * vx_scalar, capped at MAX_VOL_SCALAR.

    vol_scalar = VOL_TARGET / realized_vol(20d, annualized)
    vx_scalar  = min(VX_NEUTRAL / vx, 1.0), floor at VX_FLOOR_SCALAR
    """
    # Realized volatility (trailing VOL_LOOKBACK days, annualized)
    rv = ret_1d.rolling(VOL_LOOKBACK, min_periods=10).std() * np.sqrt(252)
    rv = rv.clip(lower=MIN_REALVOL)
    rv = rv.reindex(index).ffill().bfill()

    vol_scalar = (VOL_TARGET / rv).clip(upper=MAX_VOL_SCALAR)

    # VX overlay: scale down when market is stressed
    vx = vx_daily.reindex(index).ffill().bfill()
    # VX <= VX_NEUTRAL -> scalar = 1.0; higher VX -> linear decrease
    vx_scalar = (VX_NEUTRAL / vx.clip(lower=VX_NEUTRAL)).clip(
        lower=VX_FLOOR_SCALAR, upper=1.0)

    combined = (vol_scalar * vx_scalar).clip(upper=MAX_VOL_SCALAR)
    return combined.rename('size_scalar')


# ─────────────────────────────────────────────
# 2. FINANCIAL BACKTEST
# ─────────────────────────────────────────────
def financial_backtest(pos: pd.Series, target_10d: pd.Series,
                       size_scalar: pd.Series,
                       initial_capital: float = INITIAL_CAPITAL) -> dict:
    """
    pos:          ±1 / 0 position Series
    target_10d:   realized 10-day log return (the model target)
    size_scalar:  vol & VX combined multiplier
    initial_capital: starting equity in USD

    Returns dict with equity curve, drawdown, costs, trade stats.
    """
    # Align all series to pos.index
    tgt   = target_10d.reindex(pos.index).fillna(0)
    sz    = size_scalar.reindex(pos.index).fillna(1.0)
    pos_s = pos.shift(1).fillna(0)          # previous day position

    # ── Sized raw daily PnL (as % of capital) ──────────────────────────
    # raw  = pos * target_10d / FORWARD_DAYS  (unlevered daily log-ret)
    # sized = sz * raw  (vol-targeted position size applied)
    raw_pnl   = pos * tgt / FORWARD_DAYS
    sized_pnl = sz * raw_pnl

    # ── Transaction costs (% of capital, applied on position changes) ──
    # Size of previous position (needed for close/reversal cost)
    sz_prev = sz.shift(1).fillna(0)

    entry_mask   = (pos_s == 0) & (pos != 0)              # flat -> active
    exit_mask    = (pos_s != 0) & (pos == 0)              # active -> flat
    reverse_mask = (pos_s != 0) & (pos != 0) & (pos != pos_s)  # flip sign

    cost_pct = pd.Series(0.0, index=pos.index)
    cost_pct[entry_mask]   = COST_OPEN  * sz[entry_mask].abs()
    cost_pct[exit_mask]    = COST_CLOSE * sz_prev[exit_mask].abs()
    cost_pct[reverse_mask] = (COST_CLOSE * sz_prev[reverse_mask].abs() +
                               COST_OPEN  * sz[reverse_mask].abs())

    # ── Net daily return ───────────────────────────────────────────────
    net_daily = sized_pnl - cost_pct

    # ── Compounding equity curve ───────────────────────────────────────
    growth = (1.0 + net_daily).cumprod()
    equity = initial_capital * growth

    # ── Dollar-denominated metrics ─────────────────────────────────────
    prev_equity = np.concatenate([[initial_capital], equity.iloc[:-1].values])
    dollar_pnl  = pd.Series(prev_equity * net_daily.values, index=pos.index)
    cost_dollar = pd.Series(prev_equity * cost_pct.values,  index=pos.index)

    # ── Drawdown ───────────────────────────────────────────────────────
    peak           = equity.cummax()
    underwater_pct = (equity - peak) / peak * 100.0
    max_dd_dollar  = (equity - peak).min()
    max_dd_pct     = underwater_pct.min()

    # ── Recovery Factor ────────────────────────────────────────────────
    net_profit      = equity.iloc[-1] - initial_capital
    recovery_factor = (net_profit / abs(max_dd_dollar)
                       if max_dd_dollar < 0 else float('inf'))

    # ── Annualized Sharpe (on sized daily returns) ─────────────────────
    sharpe = sized_pnl.mean() / (sized_pnl.std() + eps) * np.sqrt(252)

    # ── Trade-level stats ──────────────────────────────────────────────
    # A "trade" = run of consecutive days with the same non-zero direction
    direction_change = (pos != pos.shift(1)).astype(int).cumsum()
    active_mask      = pos != 0
    if active_mask.any():
        trade_id         = direction_change[active_mask]
        trade_pnl_dollar = dollar_pnl[active_mask].groupby(trade_id).sum()
        avg_trade_dollar = trade_pnl_dollar.mean()
        n_trades         = len(trade_pnl_dollar)
        gross_wins       = trade_pnl_dollar.clip(lower=0).sum()
        gross_losses     = (-trade_pnl_dollar).clip(lower=0).sum()
        profit_factor    = gross_wins / (gross_losses + eps)
    else:
        avg_trade_dollar = 0.0
        n_trades         = 0
        gross_wins       = 0.0
        gross_losses     = 0.0
        profit_factor    = 0.0

    return dict(
        equity         = equity,
        dollar_pnl     = dollar_pnl,
        cost_dollar    = cost_dollar,
        sized_pnl      = sized_pnl,
        sz             = sz,
        underwater_pct = underwater_pct,
        max_dd_dollar  = max_dd_dollar,
        max_dd_pct     = max_dd_pct,
        net_profit     = net_profit,
        recovery_factor= recovery_factor,
        sharpe         = sharpe,
        total_cost_dollar = cost_dollar.sum(),
        avg_trade_dollar  = avg_trade_dollar,
        n_trades          = n_trades,
        profit_factor     = profit_factor,
        gross_wins        = gross_wins,
        gross_losses      = gross_losses,
    )


# ─────────────────────────────────────────────
# 3. PRINT AUDIT TABLE
# ─────────────────────────────────────────────
def print_audit(name: str,
                fb_val: dict, fb_oos: dict,
                equity_full: pd.Series):
    """Print financial summary for one strategy."""
    final_val = fb_val['equity'].iloc[-1]
    final_oos = fb_oos['equity'].iloc[-1]
    # Combined max drawdown
    peak_full = equity_full.cummax()
    max_dd_full_pct = ((equity_full - peak_full) / peak_full * 100).min()

    print(f"\n  [{name}]")
    print(f"    VAL period (2021-2024):")
    print(f"      Start capital:   ${INITIAL_CAPITAL:>10,.0f}")
    print(f"      End capital:     ${final_val:>10,.0f}   (net +${fb_val['net_profit']:+,.0f})")
    print(f"      Max DD:          ${fb_val['max_dd_dollar']:>10,.0f}   ({fb_val['max_dd_pct']:.2f}%)")
    print(f"      Sharpe:          {fb_val['sharpe']:>10.3f}")
    print(f"      Total costs:     ${fb_val['total_cost_dollar']:>10,.0f}")
    print(f"      Avg trade P&L:   ${fb_val['avg_trade_dollar']:>10,.0f}   ({fb_val['n_trades']} trades)")
    print(f"      Profit factor:   {fb_val['profit_factor']:>10.2f}   "
          f"(wins ${fb_val['gross_wins']:,.0f} / losses ${fb_val['gross_losses']:,.0f})")

    print(f"    OOS period (2024-2026):")
    print(f"      Start capital:   ${final_val:>10,.0f}")
    print(f"      End capital:     ${final_oos:>10,.0f}   (net +${fb_oos['net_profit']:+,.0f})")
    print(f"      Max DD:          ${fb_oos['max_dd_dollar']:>10,.0f}   ({fb_oos['max_dd_pct']:.2f}%)")
    print(f"      Recovery Factor: {fb_oos['recovery_factor']:>10.2f}"
          f"   {'EXCELLENT' if fb_oos['recovery_factor'] >= 2.0 else 'OK' if fb_oos['recovery_factor'] >= 1.0 else 'POOR'}")
    print(f"      Sharpe:          {fb_oos['sharpe']:>10.3f}")
    print(f"      Total costs:     ${fb_oos['total_cost_dollar']:>10,.0f}")
    print(f"      Avg trade P&L:   ${fb_oos['avg_trade_dollar']:>10,.0f}   ({fb_oos['n_trades']} trades)")
    print(f"      Profit factor:   {fb_oos['profit_factor']:>10.2f}")

    print(f"    COMBINED (2021-2026):")
    print(f"      Final equity:    ${final_oos:>10,.0f}   (total +${final_oos-INITIAL_CAPITAL:+,.0f})")
    print(f"      Max DD (full):   {max_dd_full_pct:.2f}%")
    print(f"      Total costs:     ${fb_val['total_cost_dollar']+fb_oos['total_cost_dollar']:>10,.0f}")


def print_comparison_table(results: dict):
    """Summary comparison table for all strategies."""
    print('\n' + '='*90)
    print('FINANCIAL AUDIT -- COMPARISON TABLE ($100,000 initial capital)')
    print('='*90)
    header = (f"{'Strategy':<20} {'End_Val($)':>11} {'NetProfit_V':>11} {'MaxDD_V($)':>11} "
              f"{'RF_OOS':>7} {'End_OOS($)':>11} {'NetProfit_O':>11} {'MaxDD_O($)':>10} {'Costs_Tot':>10}")
    print(header)
    print('-'*90)
    for name, (fb_v, fb_o) in results.items():
        end_v = fb_v['equity'].iloc[-1]
        end_o = fb_o['equity'].iloc[-1]
        rf_mark = '*' if fb_o['recovery_factor'] >= 2.0 else ' '
        print(f"{name:<20} "
              f"${end_v:>10,.0f} "
              f"${fb_v['net_profit']:>+10,.0f} "
              f"${fb_v['max_dd_dollar']:>10,.0f} "
              f"{fb_o['recovery_factor']:>6.2f}{rf_mark} "
              f"${end_o:>10,.0f} "
              f"${fb_o['net_profit']:>+10,.0f} "
              f"${fb_o['max_dd_dollar']:>9,.0f} "
              f"${fb_v['total_cost_dollar']+fb_o['total_cost_dollar']:>9,.0f}")
    print('-'*90)
    print('  RF = Recovery Factor (Net OOS Profit / Max OOS Drawdown $).  * = RF >= 2.0 (excellent)')
    print('='*90)


# ─────────────────────────────────────────────
# 4. CHARTS
# ─────────────────────────────────────────────
def plot_equity_and_drawdown(all_results: dict, vx_daily: pd.Series):
    """
    Chart A: Equity curves in dollars (all 4 strategies, VAL+OOS)
    Chart B: Underwater drawdown (%) for best strategy (Ensemble)
    Chart C: Position size (size scalar over time) + VX overlay
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                              gridspec_kw={'height_ratios': [3, 1.5, 1.2]})
    fig.suptitle(
        'Financial Audit: Monthly WFO | Vol-Targeting 10% | $100,000 Capital\n'
        'Transaction costs: 0.03% open / 0.10% close | VX-adjusted position sizing',
        fontsize=11, y=0.995)

    oos_ts = pd.Timestamp(OOS_START)

    # ── Panel A: Equity curves ─────────────────────────────────────────
    ax = axes[0]
    for name, (fb_v, fb_o) in all_results.items():
        c = COLORS[name]
        eq_full = pd.concat([fb_v['equity'], fb_o['equity']])
        ax.plot(eq_full.index, eq_full.values / 1_000,
                color=c, lw=2.0 if name == 'Ensemble' else 1.4,
                label=f"{name}  ${fb_o['equity'].iloc[-1]:,.0f}")

    ax.axhline(INITIAL_CAPITAL / 1_000, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(oos_ts, color='red', lw=1.5, ls=':', label='OOS start')
    ax.fill_betweenx([0, 10_000], oos_ts, pd.Timestamp(OOS_END),
                     color='red', alpha=0.04)
    ax.set_ylabel('Account Equity ($000)', fontsize=10)
    ax.set_title('Equity Curve: $100,000 -> Final Balance (VAL solid | OOS shaded)', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
    ax.set_ylim(bottom=50)   # guard against extreme outliers

    # ── Panel B: Underwater drawdown (Ensemble) ────────────────────────
    ax2 = axes[1]
    fb_v_ens = all_results['Ensemble'][0]
    fb_o_ens = all_results['Ensemble'][1]
    uw_full = pd.concat([fb_v_ens['underwater_pct'], fb_o_ens['underwater_pct']])

    ax2.fill_between(uw_full.index, uw_full.values, 0,
                     where=uw_full.values < 0,
                     color='#E74C3C', alpha=0.55, label='Drawdown (Ensemble)')
    ax2.plot(uw_full.index, uw_full.values, color='#C0392B', lw=1.0, alpha=0.8)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.axvline(oos_ts, color='red', lw=1.5, ls=':')
    ax2.fill_betweenx([-100, 0], oos_ts, pd.Timestamp(OOS_END),
                      color='red', alpha=0.04)
    ax2.set_ylabel('Underwater Drawdown (%)', fontsize=10)
    ax2.set_title(f'Drawdown from Peak — Ensemble Strategy'
                  f'  (VAL max={fb_v_ens["max_dd_pct"]:.1f}%'
                  f'  OOS max={fb_o_ens["max_dd_pct"]:.1f}%)', fontsize=10)
    ax2.legend(fontsize=8.5, loc='lower left')
    ax2.grid(alpha=0.25)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    # ── Panel C: Size scalar + VX ──────────────────────────────────────
    ax3 = axes[2]
    ax3_r = ax3.twinx()

    sz_full = pd.concat([fb_v_ens['sz'], fb_o_ens['sz']])
    # daily smoothed size (15d MA for readability)
    sz_ma = sz_full.rolling(15, min_periods=5).mean()

    ax3.fill_between(sz_ma.index, sz_ma.values, 0,
                     where=sz_ma.values > 0, color='#27AE60', alpha=0.35,
                     label='Avg position size (15d MA)')
    ax3.plot(sz_ma.index, sz_ma.values, color='#1E8449', lw=1.2)
    ax3.axhline(1.0, color='#1E8449', lw=0.6, ls='--', alpha=0.5)
    ax3.axvline(oos_ts, color='red', lw=1.5, ls=':')
    ax3.set_ylabel('Size Scalar (×)', fontsize=9)

    # VX overlay
    vx_plot = vx_daily.reindex(sz_full.index).ffill()
    ax3_r.plot(vx_plot.index, vx_plot.values, color='#E74C3C',
               lw=0.9, alpha=0.6, label='VX (rhs)')
    ax3_r.axhline(VX_NEUTRAL, color='#E74C3C', lw=0.6, ls='--', alpha=0.4)
    ax3_r.set_ylabel('VX Level', fontsize=9, color='#C0392B')
    ax3_r.tick_params(axis='y', colors='#C0392B')

    ax3.set_title('Volatility-Targeted Position Size (green) vs VX Stress Level (red)', fontsize=10)
    lines_a, labs_a = ax3.get_legend_handles_labels()
    lines_b, labs_b = ax3_r.get_legend_handles_labels()
    ax3.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8, loc='upper right')
    ax3.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig('risk_equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved -> risk_equity_curve.png')


def plot_trade_log(all_results: dict):
    """Bar chart: trade stats comparison across strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Trade Log Summary — Financial Metrics per Strategy', fontsize=12)

    names = list(all_results.keys())
    cols  = [COLORS[n] for n in names]
    x     = np.arange(len(names))

    def bar_panel(ax, values, title, fmt='${:.0f}', ylabel='', *, baseline=0):
        bars = ax.bar(x, values, color=cols, alpha=0.8, edgecolor='white', lw=1.5)
        ax.axhline(baseline, color='k', lw=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (max(values) - min(values)) * 0.02 * np.sign(v) if v else 0.01,
                    fmt.format(v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8.5)

    # Panel 1: Avg trade P&L (combined VAL+OOS)
    avg_pnls = []
    for name, (fb_v, fb_o) in all_results.items():
        # weighted average by n_trades
        nv, no = fb_v['n_trades'], fb_o['n_trades']
        if nv + no > 0:
            avg = (fb_v['avg_trade_dollar'] * nv + fb_o['avg_trade_dollar'] * no) / (nv + no)
        else:
            avg = 0.0
        avg_pnls.append(avg)
    bar_panel(axes[0], avg_pnls, 'Avg Profit per Trade (VAL+OOS)',
              fmt='${:+,.0f}', ylabel='USD / trade')

    # Panel 2: Total commissions paid (VAL+OOS combined)
    total_costs = [fb_v['total_cost_dollar'] + fb_o['total_cost_dollar']
                   for fb_v, fb_o in all_results.values()]
    bar_panel(axes[1], total_costs, 'Total Commissions Paid (VAL+OOS)',
              fmt='${:,.0f}', ylabel='USD')

    # Panel 3: Profit factor (OOS only — more relevant)
    pf_oos = [fb_o['profit_factor'] for _, fb_o in all_results.values()]
    bars3 = axes[2].bar(x, pf_oos, color=cols, alpha=0.8, edgecolor='white', lw=1.5)
    axes[2].axhline(1.0, color='red', lw=1.0, ls='--', label='Breakeven (1.0)')
    axes[2].axhline(1.5, color='orange', lw=0.8, ls=':', label='Good (1.5)')
    axes[2].set_title('Profit Factor — OOS 2024-2026\n(Gross Wins / Gross Losses)', fontsize=10)
    axes[2].set_ylabel('Profit Factor', fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15, fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].legend(fontsize=8)
    for bar, v in zip(bars3, pf_oos):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('risk_trade_log.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved -> risk_trade_log.png')


def plot_recovery_summary(all_results: dict):
    """Recovery Factor + Sharpe scatter for OOS."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('OOS Risk/Reward Summary (2024-2026)', fontsize=12)

    names = list(all_results.keys())
    cols  = [COLORS[n] for n in names]
    x     = np.arange(len(names))

    # Recovery Factor
    rfs = [min(fb_o['recovery_factor'], 10.0)  # cap at 10 for display
           for _, fb_o in all_results.values()]
    bars = axes[0].bar(x, rfs, color=cols, alpha=0.8, edgecolor='white', lw=1.5)
    axes[0].axhline(2.0, color='green', lw=1.2, ls='--', label='Excellent >= 2.0')
    axes[0].axhline(1.0, color='orange', lw=1.0, ls=':', label='OK >= 1.0')
    axes[0].set_title('Recovery Factor (OOS)\nNet Profit / Max Drawdown $', fontsize=10)
    axes[0].set_ylabel('Recovery Factor', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8)
    for bar, rf, (_, fb_o) in zip(bars, rfs, all_results.values()):
        label = (f'{fb_o["recovery_factor"]:.2f}' if fb_o['recovery_factor'] < 10
                 else '>10')
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     rf + 0.05, label, ha='center', va='bottom', fontsize=9)

    # Sharpe OOS
    sharpes_oos = [fb_o['sharpe'] for _, fb_o in all_results.values()]
    bars2 = axes[1].bar(x, sharpes_oos, color=cols, alpha=0.8, edgecolor='white', lw=1.5)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].axhline(1.0, color='green', lw=1.0, ls='--', label='Sharpe >= 1.0 (good)')
    axes[1].set_title('Annualized Sharpe Ratio (OOS)\nVol-targeted returns', fontsize=10)
    axes[1].set_ylabel('Sharpe', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=8)
    for bar, v in zip(bars2, sharpes_oos):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.03 * np.sign(v) if v else 0.03,
                     f'{v:+.3f}', ha='center',
                     va='bottom' if v >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig('risk_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved -> risk_recovery.png')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()

    print('=' * 72)
    print('RISK AUDIT & MONEY MANAGEMENT -- Monthly WFO Financial Analysis')
    print('=' * 72)
    print(f'  Capital: ${INITIAL_CAPITAL:,}  |  Vol-target: {VOL_TARGET*100:.0f}%/yr'
          f'  |  Lookback: {VOL_LOOKBACK}d')
    print(f'  Costs: open={COST_OPEN*100:.2f}%  close={COST_CLOSE*100:.2f}%')
    print(f'  VX overlay: neutral={VX_NEUTRAL:.0f}  floor={VX_FLOOR_SCALAR:.2f}x')

    # ── Step 1: Load data ──────────────────────────────────────────────
    print('\nStep 1 -- Loading data & building features...')
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool = json.load(f)
    pool_names = [r['name'] for r in pool]
    source_map = {r['name']: r.get('source', 'single_asset') for r in pool}

    full    = load_all_aligned()
    feat_df = build_feature_matrix(full, pool_names)
    close   = full['es_close']
    ret_1d  = log_ret(close)
    target  = np.log(close.shift(-FORWARD_DAYS) / close)
    vx_daily = full['vx_close']
    print(f'  Features: {feat_df.shape}  |  '
          f'Data: {full.index[0].date()} -> {full.index[-1].date()}')

    # ── Step 2: Factor selection ───────────────────────────────────────
    print('\nStep 2 -- Stable factor selection...')
    top_factors = select_stable_factors(pool, feat_df, target)

    # ── Step 3: Monthly WFO ────────────────────────────────────────────
    print(f'\nStep 3 -- Monthly WFO VAL ({WFO_START} -> {WFO_END})...')
    wfo_val = run_monthly_wfo(feat_df, target, top_factors, WFO_START, WFO_END)

    print(f'\nStep 4 -- Monthly WFO OOS ({OOS_START} -> {OOS_END})...')
    wfo_oos = run_monthly_wfo(feat_df, target, top_factors, OOS_START, OOS_END)

    # ── Step 4: Vol-targeting scalars ──────────────────────────────────
    print('\nStep 5 -- Computing vol-targeting scalars...')
    size_val = compute_size_scalar(ret_1d, vx_daily, wfo_val.index)
    size_oos = compute_size_scalar(ret_1d, vx_daily, wfo_oos.index)

    print(f'  VAL  size scalar: mean={size_val.mean():.2f}  '
          f'min={size_val.min():.2f}  max={size_val.max():.2f}')
    print(f'  OOS  size scalar: mean={size_oos.mean():.2f}  '
          f'min={size_oos.min():.2f}  max={size_oos.max():.2f}')

    # ── Step 5: Financial backtests for all strategies ─────────────────
    print('\nStep 6 -- Running financial backtests...')
    all_results = {}  # name -> (fb_val, fb_oos)

    for name, (col, regime_exit) in STRATEGIES.items():
        pos_val = make_positions(wfo_val, col, regime_exit)
        pos_oos = make_positions(wfo_oos, col, regime_exit)

        fb_v = financial_backtest(pos_val, wfo_val['target'], size_val,
                                  initial_capital=INITIAL_CAPITAL)
        fb_o = financial_backtest(pos_oos, wfo_oos['target'], size_oos,
                                  initial_capital=fb_v['equity'].iloc[-1])

        all_results[name] = (fb_v, fb_o)
        equity_full = pd.concat([fb_v['equity'], fb_o['equity']])

    # ── Step 6: Print results ──────────────────────────────────────────
    print_comparison_table(all_results)

    print('\nDetailed breakdown per strategy:')
    for name, (fb_v, fb_o) in all_results.items():
        equity_full = pd.concat([fb_v['equity'], fb_o['equity']])
        print_audit(name, fb_v, fb_o, equity_full)

    # ── Step 7: Charts ─────────────────────────────────────────────────
    print('\nStep 7 -- Generating charts...')
    plot_equity_and_drawdown(all_results, vx_daily)
    plot_trade_log(all_results)
    plot_recovery_summary(all_results)

    # ── Final summary ──────────────────────────────────────────────────
    ens_v, ens_o = all_results['Ensemble']
    print('\n' + '='*72)
    print('BOTTOM LINE (Ensemble Strategy):')
    print(f'  Start: ${INITIAL_CAPITAL:,}  (Jan 2021)')
    print(f'  End VAL: ${ens_v["equity"].iloc[-1]:,.0f}  (Dec 2023)')
    print(f'  End OOS: ${ens_o["equity"].iloc[-1]:,.0f}  (Mar 2026)  <-- clean, after costs')
    print(f'  OOS Recovery Factor: {ens_o["recovery_factor"]:.2f}  '
          f'(>= 2.0 = excellent, >= 1.0 = OK)')
    print(f'  OOS Max Drawdown:    ${ens_o["max_dd_dollar"]:,.0f}  ({ens_o["max_dd_pct"]:.2f}%)')
    print(f'  Total costs paid:    ${ens_v["total_cost_dollar"]+ens_o["total_cost_dollar"]:,.0f}')
    print('='*72)

    print(f'\nTotal elapsed: {time.time()-t0:.0f}s')
    print('Saved: risk_equity_curve.png  risk_trade_log.png  risk_recovery.png')
