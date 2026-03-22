"""
meta_arbitrator.py — Portfolio Meta-Model ("The Arbitrator")
=============================================================
Step 0: ES Recovery with 10d horizon (original alpha_pool_final.json pool)
Step 1: Load 6-asset WFO predictions (ES/NQ/RTY/FDAX/FESX/CL)
Step 2: Build monthly meta-features per asset
Step 3: Walk-forward Ridge vs CatBoost meta-model
Step 4: Weight allocation (total leverage 4x, per-asset 0.3x–1.5x)
Step 5: Dollar backtest $100k capital
Step 6: Charts (equity, weights-over-time, metrics table)
"""

import os, sys, json, warnings, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# ── Imports from existing modules ─────────────────────────────────────────────
from generate_intermarket import load_all_aligned, log_ret
from monthly_wfo import (
    build_feature_matrix as build_feature_matrix_es,
    select_stable_factors,
    rolling_wfo,
    FORWARD_DAYS as ORIG_FORWARD_DAYS,
    TOP_N_FACTORS, Q_LONG, Q_SHORT, NOISE_PCTILE,
    WFO_START, WFO_END, OOS_START, OOS_END,
    LGB_PARAMS, LGB_ITER, CAT_PARAMS,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR      = 'results'
META_DIR         = os.path.join(RESULTS_DIR, 'meta')
os.makedirs(META_DIR, exist_ok=True)

# 6-asset portfolio (E6 excluded)
ASSETS = ['ES', 'NQ', 'RTY', 'FDAX', 'FESX', 'CL']
ASSET_HORIZONS = {
    'ES':   10,    # recovered 10d
    'NQ':   15,
    'RTY':  20,
    'FDAX': 10,
    'FESX': 10,
    'CL':   10,
}

# Capital & leverage
INITIAL_CAPITAL  = 100_000
TOTAL_LEVERAGE   = 4.0
MIN_WEIGHT       = 0.3
MAX_WEIGHT       = 1.2
EQ_WEIGHT        = TOTAL_LEVERAGE / len(ASSETS)   # 0.667

# Commission  — realistic micro-futures level
# $1.50 RT per contract × 1 MES contract on $25k notional = 0.006% RT
COMM_RATE_RT     = 0.00006   # 0.006% round-trip (was 0.03% — 5x too high)

# Meta-model
META_MIN_TRAIN_WINDOWS = 12  # minimum 12 months before first prediction
RIDGE_ALPHA      = 10.0
CAT_META_PARAMS  = dict(
    iterations=100, learning_rate=0.03, depth=3, l2_leaf_reg=50,
    random_seed=42, allow_writing_files=False, verbose=False,
)

COLORS = {
    'ES':   '#2980B9', 'NQ':   '#E74C3C', 'RTY':  '#27AE60',
    'FDAX': '#8E44AD', 'FESX': '#F39C12', 'CL':   '#D35400',
}
eps = 1e-8

FORCE_ES_RERUN = False   # set True to force ES 10d rerun


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — ES RECOVERY (10d, original alpha_pool_final.json)
# ══════════════════════════════════════════════════════════════════════════════

def run_es_10d():
    """
    Rebuilds ES WFO with 10d horizon using the original alpha_pool_final.json
    and generate_intermarket.load_all_aligned() (ES/NQ/GC/VX/ZN/DX partners).
    Saves to results/ES/wfo_val_10d.pkl and wfo_oos_10d.pkl.
    """
    val_path = os.path.join(RESULTS_DIR, 'ES', 'wfo_val_10d.pkl')
    oos_path = os.path.join(RESULTS_DIR, 'ES', 'wfo_oos_10d.pkl')

    if not FORCE_ES_RERUN and os.path.exists(val_path) and os.path.exists(oos_path):
        print('  [ES 10d] Loading cached WFO results...')
        return pd.read_pickle(val_path), pd.read_pickle(oos_path)

    print('  [ES 10d] Loading data via load_all_aligned()...')
    full    = load_all_aligned()
    close   = full['es_close']
    target  = np.log(close.shift(-10) / close)   # 10d forward return

    print('  [ES 10d] Loading alpha_pool_final.json (350 factors, 10d)...')
    with open('alpha_pool_final.json', 'r') as f:
        pool = json.load(f)
    pool_names = [p['name'] for p in pool]
    source_map = {p['name']: p.get('source', 'single_asset') for p in pool}

    print('  [ES 10d] Building feature matrix...')
    feat_path_10d = os.path.join(RESULTS_DIR, 'ES', 'features_10d.pkl')
    if not FORCE_ES_RERUN and os.path.exists(feat_path_10d):
        feat_df = pd.read_pickle(feat_path_10d)
        print(f'  [ES 10d] Loaded cached features {feat_df.shape}')
    else:
        feat_df = build_feature_matrix_es(full, pool_names)
        feat_df.to_pickle(feat_path_10d)
        print(f'  [ES 10d] Feature matrix: {feat_df.shape}')

    print('  [ES 10d] Selecting stable factors...')
    top_factors = select_stable_factors(pool, feat_df, target)
    n_im = sum(1 for n in top_factors if source_map.get(n, '') == 'intermarket')
    print(f'  [ES 10d] SA={len(top_factors)-n_im}  IM={n_im}  top-5: {top_factors[:5]}')

    print(f'  [ES 10d] Monthly WFO — VAL ({WFO_START} -> {WFO_END})')
    wfo_val = rolling_wfo(feat_df, target, top_factors, WFO_START, WFO_END)
    wfo_val.to_pickle(val_path)

    print(f'  [ES 10d] Monthly WFO — OOS ({OOS_START} -> {OOS_END})')
    wfo_oos = rolling_wfo(feat_df, target, top_factors, OOS_START, OOS_END)
    wfo_oos.to_pickle(oos_path)

    print(f'  [ES 10d] Done. Saved to results/ES/')
    return wfo_val, wfo_oos


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL ASSETS
# ══════════════════════════════════════════════════════════════════════════════

def load_asset_wfo(ticker: str) -> tuple:
    """Load wfo_val and wfo_oos for a given asset."""
    d = os.path.join(RESULTS_DIR, ticker)
    # ES uses the 10d-specific files
    suffix = '_10d' if ticker == 'ES' else ''
    wfo_val = pd.read_pickle(os.path.join(d, f'wfo_val{suffix}.pkl'))
    wfo_oos = pd.read_pickle(os.path.join(d, f'wfo_oos{suffix}.pkl'))
    return wfo_val, wfo_oos


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RECONSTRUCT POSITIONS & PnL
# ══════════════════════════════════════════════════════════════════════════════

def make_positions_re(wfo: pd.DataFrame, regime_exit: bool = True) -> pd.Series:
    pos = pd.Series(0.0, index=wfo.index)
    for win_id in wfo['window'].unique():
        m      = wfo['window'] == win_id
        sig    = wfo.loc[m, 'p_ens']
        noise  = wfo.loc[m, 'noise'].iloc[0]
        thr_hi = sig.quantile(Q_LONG)
        thr_lo = sig.quantile(Q_SHORT)
        lm     = m & (wfo['p_ens'] >= thr_hi)
        sm     = m & (wfo['p_ens'] <= thr_lo)
        if regime_exit:
            noisy = (wfo['p_lgb'].abs() < noise) & (wfo['p_cat'].abs() < noise)
            lm    = lm & ~noisy
            sm    = sm & ~noisy
        pos[lm] =  1.0
        pos[sm] = -1.0
    return pos


def daily_pnl_series(wfo: pd.DataFrame, pos: pd.Series, horizon: int) -> pd.Series:
    """Daily log-return P&L (unlevered, per unit capital)."""
    tgt  = wfo['target'].fillna(0)
    raw  = pos * tgt / horizon
    chg  = pos.diff().abs().fillna(0) > 0
    return raw - chg.astype(float) * COMM_RATE_RT


def asset_sharpe(pnl: pd.Series) -> float:
    return float(pnl.mean() / (pnl.std() + eps) * np.sqrt(252))


def profit_factor(pnl: pd.Series) -> float:
    pos_sum = pnl[pnl > 0].sum()
    neg_sum = abs(pnl[pnl < 0].sum())
    return float(pos_sum / (neg_sum + eps))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — META-FEATURES BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_meta_dataset(wfo_data: dict, pos_data: dict, pnl_data: dict) -> pd.DataFrame:
    """
    For each calendar month (WFO window start), compute per-asset features:
      pred_mean, pred_std, ic_curr, sharpe_1mo, sharpe_3mo,
      vol_3mo, pf_3mo, cum_ret_1mo, ic_3mo, active_ratio
    Returns DataFrame indexed by month start date, columns = {ticker}_{feature}.
    Also adds TARGET columns: {ticker}_target_sharpe (next-month realized Sharpe).
    """
    tickers = list(wfo_data.keys())

    # Collect all WFO row assignments: date -> (ticker, window_id)
    # Use the first asset to define the monthly grid
    ref_ticker = tickers[0]
    ref_wfo    = wfo_data[ref_ticker]

    # Get window start dates
    win_starts = {}
    for t in tickers:
        wfo = wfo_data[t]
        for win_id in sorted(wfo['window'].unique()):
            m    = wfo['window'] == win_id
            date = wfo[m].index[0]
            win_starts.setdefault(win_id, {})[t] = date

    # Align by window using ref_ticker
    ref_windows = sorted(ref_wfo['window'].unique())

    rows = []
    for w_idx, win_id in enumerate(ref_windows):
        row = {'win_id': win_id}

        for ticker in tickers:
            wfo  = wfo_data[ticker]
            pos  = pos_data[ticker]
            pnl  = pnl_data[ticker]
            h    = ASSET_HORIZONS[ticker]

            # Current window data
            wfo_wins = sorted(wfo['window'].unique())
            if win_id not in wfo_wins:
                # use closest window
                closest = min(wfo_wins, key=lambda w: abs(w - win_id))
                m_cur = wfo['window'] == closest
            else:
                m_cur = wfo['window'] == win_id

            cur_pens   = wfo.loc[m_cur, 'p_ens']
            cur_tgt    = wfo.loc[m_cur, 'target']
            cur_idx    = wfo[m_cur].index

            # IC current window
            vm = cur_pens.notna() & cur_tgt.notna()
            ic_curr = spearmanr(cur_pens[vm], cur_tgt[vm])[0] if vm.sum() >= 5 else 0.0

            # Active ratio
            cur_pos     = pos.reindex(cur_idx).fillna(0)
            active_ratio = (cur_pos != 0).mean()

            # Historical PnL (last 1mo and 3mo)
            cur_start = cur_idx[0] if len(cur_idx) > 0 else pd.Timestamp('2020-01-01')
            pnl_hist = pnl[pnl.index < cur_start]
            pnl_1mo  = pnl_hist.iloc[-21:]   if len(pnl_hist) >= 21  else pnl_hist
            pnl_3mo  = pnl_hist.iloc[-63:]   if len(pnl_hist) >= 63  else pnl_hist

            sh1   = asset_sharpe(pnl_1mo) if len(pnl_1mo) >= 5 else 0.0
            sh3   = asset_sharpe(pnl_3mo) if len(pnl_3mo) >= 5 else 0.0
            vol3  = float(pnl_3mo.std() * np.sqrt(252)) if len(pnl_3mo) >= 5 else 0.01
            pf3   = profit_factor(pnl_3mo) if len(pnl_3mo) >= 5 else 1.0
            cr1   = float(pnl_1mo.sum()) if len(pnl_1mo) > 0 else 0.0

            # IC over last 3 windows
            prev_wins = wfo_wins[:max(0, wfo_wins.index(win_id) if win_id in wfo_wins else 0)]
            ic_hist   = []
            for pw in prev_wins[-3:]:
                mm = wfo['window'] == pw
                pp = wfo.loc[mm, 'p_ens']; tt = wfo.loc[mm, 'target']
                vv = pp.notna() & tt.notna()
                if vv.sum() >= 5:
                    ic_hist.append(spearmanr(pp[vv], tt[vv])[0])
            ic_3mo = float(np.mean(ic_hist)) if ic_hist else 0.0

            row.update({
                f'{ticker}_pred_mean':    float(cur_pens.mean()),
                f'{ticker}_pred_std':     float(cur_pens.std()),
                f'{ticker}_ic_curr':      float(ic_curr),
                f'{ticker}_sharpe_1mo':   float(sh1),
                f'{ticker}_sharpe_3mo':   float(sh3),
                f'{ticker}_vol_3mo':      float(vol3),
                f'{ticker}_pf_3mo':       float(np.clip(pf3, 0, 5)),
                f'{ticker}_cum_ret_1mo':  float(cr1),
                f'{ticker}_ic_3mo':       float(ic_3mo),
                f'{ticker}_active_ratio': float(active_ratio),
            })

        rows.append(row)

    meta_df = pd.DataFrame(rows).set_index('win_id')

    # Add target: next-window realized Sharpe per asset
    for ticker in tickers:
        wfo  = wfo_data[ticker]
        pos  = pos_data[ticker]
        pnl  = pnl_data[ticker]
        wins = sorted(wfo['window'].unique())
        tgt_sh = {}
        for win_id in wins:
            m   = wfo['window'] == win_id
            idx = wfo[m].index
            pnl_w = pnl.reindex(idx).fillna(0)
            tgt_sh[win_id] = asset_sharpe(pnl_w)
        # shift by 1 (we predict NEXT window)
        tgt_series = pd.Series(tgt_sh)
        tgt_series.index = tgt_series.index - 1   # align: features at t → target at t+1
        meta_df[f'{ticker}_target_sharpe'] = tgt_series

    meta_df = meta_df.dropna()
    return meta_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WEIGHT ALLOCATION
# ══════════════════════════════════════════════════════════════════════════════

def scores_to_weights(scores: np.ndarray) -> np.ndarray:
    """
    Converts raw meta-scores (per-asset predicted Sharpe) to portfolio weights.
    Constraints: each weight in [MIN_WEIGHT, MAX_WEIGHT], sum = TOTAL_LEVERAGE.
    """
    scores = np.array(scores, dtype=float)
    # Shift to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm = (scores - s_min) / (s_max - s_min)
    else:
        norm = np.full(len(scores), 0.5)
    # Map to [MIN_WEIGHT, MAX_WEIGHT]
    weights = MIN_WEIGHT + norm * (MAX_WEIGHT - MIN_WEIGHT)
    # Scale to TOTAL_LEVERAGE
    weights = weights / weights.sum() * TOTAL_LEVERAGE
    # Hard clip & renormalize
    weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
    weights = weights / weights.sum() * TOTAL_LEVERAGE
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WALK-FORWARD META-MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_meta_wfo(meta_df: pd.DataFrame, val_wins: list, oos_wins: list,
                 model_type: str = 'ridge') -> dict:
    """
    Walk-forward meta-model.
    model_type: 'ridge' or 'catboost'
    Returns dict with weight time-series and equity curve.
    """
    feat_cols = [c for c in meta_df.columns if '_target_' not in c]
    tgt_cols  = [f'{t}_target_sharpe' for t in ASSETS]

    all_wins   = sorted(meta_df.index.tolist())
    all_results = {}   # win_id -> predicted weights

    def predict_weights(model, X_pred, scaler=None):
        if scaler is not None:
            X_pred = scaler.transform(X_pred)
        if isinstance(model, list):   # list of CatBoost models
            scores = np.array([m.predict(X_pred)[0] for m in model])
        else:
            scores = model.predict(X_pred)[0]
        return scores_to_weights(scores)

    # Walk-forward: train on all data up to current window, predict next
    for i, win_id in enumerate(all_wins):
        train_wins = [w for w in all_wins[:i] if w in val_wins]
        if len(train_wins) < META_MIN_TRAIN_WINDOWS:
            # Not enough data — use equal weights
            all_results[win_id] = np.full(len(ASSETS), EQ_WEIGHT)
            continue

        X_train = meta_df.loc[train_wins, feat_cols].values.astype(float)
        y_train = meta_df.loc[train_wins, tgt_cols].values.astype(float)
        X_pred  = meta_df.loc[[win_id], feat_cols].values.astype(float)

        # Handle NaN
        nan_rows = np.isnan(X_train).any(1) | np.isnan(y_train).any(1)
        X_train, y_train = X_train[~nan_rows], y_train[~nan_rows]
        if len(X_train) < 6:
            all_results[win_id] = np.full(len(ASSETS), EQ_WEIGHT)
            continue

        if model_type == 'ridge':
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_pr_s = scaler.transform(X_pred)
            model  = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_tr_s, y_train)
            scores = model.predict(X_pr_s)[0]
            all_results[win_id] = scores_to_weights(scores)

        elif model_type == 'catboost':
            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_train)
            X_pr_s  = scaler.transform(X_pred)
            models  = []
            for ti in range(len(ASSETS)):
                m = CatBoostRegressor(**CAT_META_PARAMS)
                m.fit(X_tr_s, y_train[:, ti])
                models.append(m)
            scores = np.array([m.predict(X_pr_s)[0] for m in models])
            all_results[win_id] = scores_to_weights(scores)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PORTFOLIO DOLLAR BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_backtest(wfo_data: dict, pos_data: dict, pnl_data: dict,
                       win_weights: dict, period_wins: list,
                       label: str = '') -> dict:
    """
    Applies dynamic weights to each asset's positions.
    win_weights: {win_id: np.array(6)} aligned to ASSETS order
    period_wins: list of window IDs to include
    """
    # Build daily weight series per asset
    all_idx = []
    for ticker in ASSETS:
        wfo = wfo_data[ticker]
        mask = wfo['window'].isin(period_wins)
        all_idx.extend(wfo[mask].index.tolist())
    all_idx = sorted(set(all_idx))
    idx     = pd.DatetimeIndex(all_idx)

    meta_port_pnl = pd.Series(0.0, index=idx)
    eq_port_pnl   = pd.Series(0.0, index=idx)
    weight_ts     = pd.DataFrame(index=idx, columns=ASSETS, dtype=float)

    for ti, ticker in enumerate(ASSETS):
        wfo = wfo_data[ticker]
        pos = pos_data[ticker]
        pnl = pnl_data[ticker]  # unlevered daily pnl (commission already deducted)

        # Assign weight per day based on its WFO window
        day_weight = pd.Series(np.nan, index=idx)
        day_eq_w   = EQ_WEIGHT

        for win_id in period_wins:
            m    = wfo['window'] == win_id
            days = wfo[m].index
            w    = win_weights.get(win_id, np.full(len(ASSETS), EQ_WEIGHT))[ti]
            for d in days:
                if d in day_weight.index:
                    day_weight[d] = w

        day_weight = day_weight.ffill().fillna(EQ_WEIGHT)
        weight_ts[ticker] = day_weight

        # Levered pnl per asset = pnl × weight
        # pnl already has commission from unlevered → need to scale commission by weight
        # Simple: levered_pnl = pnl × weight  (pnl is in unit-capital log-return terms)
        asset_pnl = pnl.reindex(idx).fillna(0)
        meta_port_pnl += asset_pnl * day_weight
        eq_port_pnl   += asset_pnl * day_eq_w

    # Convert to dollar equity
    meta_equity = INITIAL_CAPITAL * (1 + meta_port_pnl.cumsum())
    eq_equity   = INITIAL_CAPITAL * (1 + eq_port_pnl.cumsum())

    def metrics(pnl_s, equity_s):
        sh  = pnl_s.mean() / (pnl_s.std() + eps) * np.sqrt(252)
        cum = pnl_s.cumsum()
        mdd = float((cum - cum.cummax()).min())
        net = float(equity_s.iloc[-1] - INITIAL_CAPITAL)
        rec = abs(net / (mdd * INITIAL_CAPITAL + eps))
        pf  = profit_factor(pnl_s)
        return dict(sharpe=sh, max_dd=mdd, max_dd_pct=mdd*100,
                    net_profit=net, recovery_factor=rec, profit_factor=pf)

    return dict(
        meta_pnl=meta_port_pnl, meta_equity=meta_equity,
        eq_pnl=eq_port_pnl,     eq_equity=eq_equity,
        weight_ts=weight_ts,
        meta_metrics=metrics(meta_port_pnl, meta_equity),
        eq_metrics=metrics(eq_port_pnl, eq_equity),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(res_ridge_oos, res_cat_oos,
                 res_ridge_val, res_cat_val,
                 es_sharpe_10d_oos: float):

    fig = plt.figure(figsize=(16, 20))
    gs  = fig.add_gridspec(5, 1,
                           height_ratios=[2.6, 1.2, 1.4, 1.6, 1.4],
                           hspace=0.44)

    ax_eq  = fig.add_subplot(gs[0])   # equity curves
    ax_uw  = fig.add_subplot(gs[1])   # underwater drawdown
    ax_bar = fig.add_subplot(gs[2])   # per-asset OOS contribution
    ax_wt  = fig.add_subplot(gs[3])   # OOS weights over time (stacked bar)
    ax_tbl = fig.add_subplot(gs[4])   # metrics table

    oos_ts = pd.Timestamp(OOS_START)

    # ── Panel 1: Equity curves ────────────────────────────────────────────────
    style_map = {
        'Ridge Meta':   ('#27AE60', '-'),
        'CatBoost Meta': ('#8E44AD', '-'),
        'Equal-Weight': ('#2980B9', '--'),
    }
    for (res_v, res_o, name) in [
        (res_ridge_val, res_ridge_oos, 'Ridge Meta'),
        (res_cat_val,   res_cat_oos,   'CatBoost Meta'),
        (res_ridge_val, res_ridge_oos, 'Equal-Weight'),
    ]:
        c, ls = style_map[name]
        key   = 'meta_equity' if 'Meta' in name else 'eq_equity'
        eq_v  = res_v[key]
        eq_o  = res_o[key]
        off   = eq_v.iloc[-1] - INITIAL_CAPITAL
        m_v   = res_v['meta_metrics'] if 'Meta' in name else res_v['eq_metrics']
        m_o   = res_o['meta_metrics'] if 'Meta' in name else res_o['eq_metrics']
        lbl_o = (f"{name} OOS  Net ${m_o['net_profit']:+,.0f}  "
                 f"Sh={m_o['sharpe']:+.2f}  DD={m_o['max_dd_pct']:+.1f}%")
        ax_eq.plot(eq_v.index, eq_v / 1000, color=c, lw=2.0, ls=ls, alpha=0.55)
        ax_eq.plot((eq_o + off).index, (eq_o + off) / 1000,
                   color=c, lw=2.2, ls=ls, label=lbl_o)

    ax_eq.axvline(oos_ts, color='red', lw=1.5, ls=':', label='OOS start')
    ax_eq.axhline(INITIAL_CAPITAL / 1000, color='k', lw=0.7, ls=':')
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
    ax_eq.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax_eq.set_title(
        'Meta-Arbitrator "Batya" — Portfolio Equity  |  $100k Capital  |  '
        f'Leverage {MIN_WEIGHT}x-{MAX_WEIGHT}x, Total {TOTAL_LEVERAGE}x\n'
        f'Commission: {COMM_RATE_RT*100:.4f}% RT  |  Regime Exit: ON  |  '
        'Solid/faded = VAL 2021-2024  |  Solid = OOS 2024-2026',
        fontsize=10
    )
    ax_eq.legend(fontsize=8.5, loc='upper left', ncol=1, framealpha=0.90)
    ax_eq.grid(alpha=0.2)

    # ── Panel 2: Underwater Drawdown ─────────────────────────────────────────
    for (res_v, res_o, name, c, ls) in [
        (res_ridge_val, res_ridge_oos, 'Ridge Meta',   '#27AE60', '-'),
        (res_cat_val,   res_cat_oos,   'CatBoost Meta', '#8E44AD', '-'),
        (res_ridge_val, res_ridge_oos, 'Equal-Weight', '#2980B9', '--'),
    ]:
        key = 'meta_pnl' if 'Meta' in name else 'eq_pnl'
        pnl_v = res_v[key]
        pnl_o = res_o[key]
        pnl_all = pd.concat([pnl_v, pnl_o]).sort_index()
        cum  = pnl_all.cumsum()
        uw   = (cum - cum.cummax()) * 100
        ax_uw.plot(uw.index, uw.values, color=c, lw=1.5, ls=ls,
                   label=f"{name}  MinDD={uw.min():+.1f}%")
        if 'Ridge' in name:
            ax_uw.fill_between(uw.index, uw.values, 0,
                               where=uw.values < 0, color=c, alpha=0.12)

    ax_uw.axvline(oos_ts, color='red', lw=1.5, ls=':')
    ax_uw.axhline(0, color='k', lw=0.8)
    ax_uw.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.1f}%'))
    ax_uw.set_ylabel('Underwater DD (%)', fontsize=9)
    ax_uw.set_title('Underwater Drawdown from Equity Peak', fontsize=10)
    ax_uw.legend(fontsize=8.5, loc='lower left', ncol=3)
    ax_uw.grid(alpha=0.2)

    # ── Panel 3: Per-asset OOS contribution ──────────────────────────────────
    meta_r_pnl_per = {}
    meta_c_pnl_per = {}
    eq_pnl_per     = {}
    asset_sharpes  = {}
    asset_pfs      = {}

    idx_oos = res_ridge_oos['meta_pnl'].index
    for ticker in ASSETS:
        p  = pnl_all_[ticker].reindex(idx_oos).fillna(0)
        wr = res_ridge_oos['weight_ts'][ticker]
        wc = res_cat_oos['weight_ts'][ticker]
        meta_r_pnl_per[ticker] = float((p * wr).sum() * 100)
        meta_c_pnl_per[ticker] = float((p * wc).sum() * 100)
        eq_pnl_per[ticker]     = float((p * EQ_WEIGHT).sum() * 100)
        asset_sharpes[ticker]  = asset_sharpe(p)
        asset_pfs[ticker]      = profit_factor(p)

    x  = np.arange(len(ASSETS))
    bw = 0.26
    ax_bar.bar(x - bw, [meta_r_pnl_per[t] for t in ASSETS],
               bw, label='Ridge Meta', color='#27AE60', alpha=0.82, edgecolor='w')
    ax_bar.bar(x,      [meta_c_pnl_per[t] for t in ASSETS],
               bw, label='CatBoost Meta', color='#8E44AD', alpha=0.82, edgecolor='w')
    ax_bar.bar(x + bw, [eq_pnl_per[t] for t in ASSETS],
               bw, label='Equal-Weight', color='#2980B9', alpha=0.82, edgecolor='w')
    # annotate Sharpe above each group
    for i, ticker in enumerate(ASSETS):
        ax_bar.text(i, max(meta_r_pnl_per[ticker], meta_c_pnl_per[ticker],
                           eq_pnl_per[ticker]) + 0.3,
                    f"Sh={asset_sharpes[ticker]:+.2f}", ha='center', fontsize=7.5, fontweight='bold')
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(ASSETS, fontsize=9)
    ax_bar.axhline(0, color='k', lw=0.8)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.1f}%'))
    ax_bar.set_ylabel('OOS Contribution (%)', fontsize=9)
    ax_bar.set_title('OOS 2024-2026 Per-Asset Contribution  |  Commission 0.006% RT', fontsize=9)
    ax_bar.legend(fontsize=8, loc='upper right')
    ax_bar.grid(axis='y', alpha=0.2)

    # ── Panel 4: Ridge OOS weights over time (stacked bar) ───────────────────
    wt_monthly = res_ridge_oos['weight_ts'].resample('ME').mean()
    bottom = np.zeros(len(wt_monthly))
    for ticker in ASSETS:
        vals = wt_monthly[ticker].fillna(EQ_WEIGHT).values
        ax_wt.bar(wt_monthly.index, vals, bottom=bottom,
                  color=COLORS[ticker], alpha=0.82, width=20, label=ticker)
        bottom += vals
    ax_wt.axhline(TOTAL_LEVERAGE, color='k', lw=1.2, ls='--',
                  label=f'Total={TOTAL_LEVERAGE}x')
    ax_wt.axhline(EQ_WEIGHT, color='gray', lw=0.8, ls=':', alpha=0.7,
                  label=f'Equal={EQ_WEIGHT:.2f}x')
    ax_wt.set_ylim(0, TOTAL_LEVERAGE + 0.8)
    ax_wt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}x'))
    ax_wt.set_ylabel('Leverage per Asset', fontsize=9)
    ax_wt.set_title('Ridge Meta — OOS Monthly Leverage Allocation (stacked)', fontsize=9)
    ax_wt.legend(fontsize=8, loc='upper right', ncol=4)
    ax_wt.grid(axis='y', alpha=0.2)

    # ── Panel 5: Full metrics table ───────────────────────────────────────────
    ax_tbl.axis('off')

    # Portfolio rows
    port_headers = ['Model', 'Period', 'Net P&L', 'MaxDD%', 'Sharpe', 'RecovFact', 'ProfitFact']
    port_rows = []
    for period, (res_r, res_c) in [('VAL 2021-24', (res_ridge_val, res_cat_val)),
                                    ('OOS 2024-26', (res_ridge_oos, res_cat_oos))]:
        for name, res in [('Ridge Meta', res_r), ('CatBoost Meta', res_c), ('Equal-Weight', res_r)]:
            m = res['meta_metrics'] if 'Meta' in name else res['eq_metrics']
            port_rows.append([
                name, period,
                f"${m['net_profit']:+,.0f}",
                f"{m['max_dd_pct']:+.2f}%",
                f"{m['sharpe']:+.3f}",
                f"{m['recovery_factor']:.2f}x",
                f"{m['profit_factor']:.2f}",
            ])

    # Per-asset rows
    asset_headers = ['Asset', 'Horizon', 'OOS Net ($)', 'OOS Sharpe', 'OOS ProfFact', 'Regime', '']
    asset_rows = []
    for ticker in ASSETS:
        p_oos = pnl_all_[ticker].reindex(idx_oos).fillna(0)
        net_d = float(p_oos.sum() * INITIAL_CAPITAL)
        sh    = asset_sharpe(p_oos)
        pf    = profit_factor(p_oos)
        asset_rows.append([
            ticker,
            f"{ASSET_HORIZONS[ticker]}d",
            f"${net_d:+,.0f}",
            f"{sh:+.3f}",
            f"{pf:.2f}",
            'ON',
            '',
        ])

    # Draw two sub-tables side by side
    tbl1 = ax_tbl.table(cellText=port_rows, colLabels=port_headers,
                         bbox=[0.0, 0.0, 0.62, 1.0], cellLoc='center')
    tbl2 = ax_tbl.table(cellText=asset_rows, colLabels=asset_headers,
                         bbox=[0.64, 0.0, 0.36, 1.0], cellLoc='center')

    for tbl in [tbl1, tbl2]:
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0)
        tbl.scale(1, 1.45)
        ncols = len(tbl._cells.get((0, 0), [0]).__class__.__mro__)  # dummy
        nrows = max(r for (r, _) in tbl._cells) + 1
        ncols = max(c for (_, c) in tbl._cells) + 1
        for j in range(ncols):
            tbl[(0, j)].set_facecolor('#2C3E50')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    # Color OOS rows
    for i, row in enumerate(port_rows):
        clr = '#E8F8F5' if ('+' in row[2] and 'OOS' in row[1]) else \
              '#FDEDEC' if ('-' in row[2] and 'OOS' in row[1]) else 'white'
        for j in range(len(port_headers)):
            tbl1[(i + 1, j)].set_facecolor(clr)
    for i, row in enumerate(asset_rows):
        clr = '#E8F8F5' if '+' in row[2] else '#FDEDEC'
        for j in range(len(asset_headers)):
            tbl2[(i + 1, j)].set_facecolor(clr)

    fig.suptitle(
        f'Meta-Arbitrator "Batya"  |  {len(ASSETS)} Assets  |  '
        f'Leverage {MIN_WEIGHT}x-{MAX_WEIGHT}x per asset, {TOTAL_LEVERAGE}x total  |  '
        f'Commission {COMM_RATE_RT*100:.4f}% RT  |  Regime Exit: ON',
        fontsize=11, fontweight='bold', y=0.998
    )

    save_path = os.path.join(META_DIR, 'meta_arbitrator_report.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Global data containers (used in plot_results)
wfo_all  = {}
pos_all  = {}
pnl_all_ = {}

if __name__ == '__main__':
    t0 = time.time()
    print('=' * 70)
    print('  META-ARBITRATOR PIPELINE')
    print(f'  Assets: {ASSETS}')
    print(f'  Capital: ${INITIAL_CAPITAL:,}  Leverage: {MIN_WEIGHT}x-{MAX_WEIGHT}x  Total: {TOTAL_LEVERAGE}x')
    print('=' * 70)

    # ── Step 0: ES Recovery ──────────────────────────────────────────────────
    print('\n[Step 0] ES Recovery with 10d horizon')
    wfo_es_val, wfo_es_oos = run_es_10d()

    # Quick ES 10d metrics
    pos_es_oos   = make_positions_re(wfo_es_oos, regime_exit=True)
    pnl_es_oos   = daily_pnl_series(wfo_es_oos, pos_es_oos, 10)
    sh_es_oos    = asset_sharpe(pnl_es_oos)
    _es_joint = wfo_es_oos[['p_ens', 'target']].dropna()
    ic_es_oos, _ = spearmanr(_es_joint['p_ens'], _es_joint['target'])
    print(f'\n  [ES 10d OOS] Sharpe={sh_es_oos:+.3f}  IC={ic_es_oos:+.4f}  '
          f'Net={pnl_es_oos.sum()*INITIAL_CAPITAL:+,.0f}$')

    # ── Step 1: Load all assets ──────────────────────────────────────────────
    print('\n[Step 1] Loading WFO data for all 6 assets')
    for ticker in ASSETS:
        try:
            wv, wo = load_asset_wfo(ticker)
            wfo_all[ticker] = {'val': wv, 'oos': wo}
            print(f'  {ticker}: VAL={len(wv)} rows  OOS={len(wo)} rows')
        except Exception as e:
            print(f'  [ERROR] {ticker}: {e}')

    # ── Step 2: Reconstruct positions & PnL ─────────────────────────────────
    print('\n[Step 2] Reconstructing positions & PnL')
    for ticker in ASSETS:
        h = ASSET_HORIZONS[ticker]
        for period in ['val', 'oos']:
            wfo = wfo_all[ticker][period]
            pos = make_positions_re(wfo, regime_exit=True)
            pnl = daily_pnl_series(wfo, pos, h)
            wfo_all[ticker][f'pos_{period}'] = pos
            wfo_all[ticker][f'pnl_{period}'] = pnl

        # Merged series for meta-features
        pos_all[ticker]  = pd.concat([wfo_all[ticker]['pos_val'],
                                       wfo_all[ticker]['pos_oos']]).sort_index()
        pnl_all_[ticker] = pd.concat([wfo_all[ticker]['pnl_val'],
                                       wfo_all[ticker]['pnl_oos']]).sort_index()

    # Flat WFO dict for meta-features
    wfo_merged = {t: pd.concat([wfo_all[t]['val'], wfo_all[t]['oos']]).sort_index()
                  for t in ASSETS}
    # Re-number windows continuously (val: 1-36, oos: 37-62)
    for ticker in ASSETS:
        wfo_v = wfo_all[ticker]['val']
        wfo_o = wfo_all[ticker]['oos']
        max_val_win = wfo_v['window'].max()
        wfo_o_adj   = wfo_o.copy()
        wfo_o_adj['window'] += max_val_win
        wfo_merged[ticker] = pd.concat([wfo_v, wfo_o_adj]).sort_index()

    val_wins_es = sorted(wfo_all['ES']['val']['window'].unique())
    oos_wins_es = sorted(wfo_all['ES']['oos']['window'].unique())
    max_val     = wfo_all['ES']['val']['window'].max()
    # Adjust OOS window ids to be continuous
    val_wins = val_wins_es
    oos_wins = [w + max_val for w in oos_wins_es]

    # ── Step 3: Build meta-features ─────────────────────────────────────────
    print('\n[Step 3] Building meta-features (monthly)')
    meta_df = build_meta_dataset(wfo_merged, pos_all, pnl_all_)
    print(f'  Meta-features shape: {meta_df.shape}')
    print(f'  Windows: {len(meta_df)}  |  '
          f'VAL wins: {len([w for w in meta_df.index if w <= max_val])}  '
          f'OOS wins: {len([w for w in meta_df.index if w > max_val])}')

    # ── Step 4: Walk-forward meta-models ────────────────────────────────────
    print('\n[Step 4] Walk-forward meta-model training')
    print('  Training Ridge...')
    ridge_weights = run_meta_wfo(meta_df, val_wins, oos_wins, model_type='ridge')
    print('  Training CatBoost...')
    cat_weights   = run_meta_wfo(meta_df, val_wins, oos_wins, model_type='catboost')

    # ── Step 5: Portfolio backtests ─────────────────────────────────────────
    print('\n[Step 5] Portfolio backtests')

    # Use per-period WFO dicts
    wfo_val_dict = {t: wfo_all[t]['val'] for t in ASSETS}
    wfo_oos_dict = {t: wfo_all[t]['oos'] for t in ASSETS}
    pos_val_dict = {t: wfo_all[t]['pos_val'] for t in ASSETS}
    pos_oos_dict = {t: wfo_all[t]['pos_oos'] for t in ASSETS}
    pnl_val_dict = {t: wfo_all[t]['pnl_val'] for t in ASSETS}
    pnl_oos_dict = {t: wfo_all[t]['pnl_oos'] for t in ASSETS}

    # Map adjusted window ids back to original for each period
    ridge_w_val = {w: ridge_weights.get(w, np.full(len(ASSETS), EQ_WEIGHT)) for w in val_wins}
    ridge_w_oos = {}
    cat_w_val   = {w: cat_weights.get(w, np.full(len(ASSETS), EQ_WEIGHT)) for w in val_wins}
    cat_w_oos   = {}
    for orig_w, adj_w in zip(oos_wins_es, oos_wins):
        ridge_w_oos[orig_w] = ridge_weights.get(adj_w, np.full(len(ASSETS), EQ_WEIGHT))
        cat_w_oos[orig_w]   = cat_weights.get(adj_w, np.full(len(ASSETS), EQ_WEIGHT))

    res_ridge_val = portfolio_backtest(wfo_val_dict, pos_val_dict, pnl_val_dict,
                                        ridge_w_val, val_wins, 'Ridge VAL')
    res_ridge_oos = portfolio_backtest(wfo_oos_dict, pos_oos_dict, pnl_oos_dict,
                                        ridge_w_oos, oos_wins_es, 'Ridge OOS')
    res_cat_val   = portfolio_backtest(wfo_val_dict, pos_val_dict, pnl_val_dict,
                                        cat_w_val, val_wins, 'CatBoost VAL')
    res_cat_oos   = portfolio_backtest(wfo_oos_dict, pos_oos_dict, pnl_oos_dict,
                                        cat_w_oos, oos_wins_es, 'CatBoost OOS')

    # ── Step 6: Print metrics ────────────────────────────────────────────────
    print('\n' + '=' * 88)
    print(f'OOS METRICS TABLE (2024-2026)  |  Commission: {COMM_RATE_RT*100:.4f}% RT  |  Regime Exit: ON')
    print('=' * 88)
    hdr = f"{'Model':<22} {'Net P&L':>10} {'MaxDD%':>8} {'Sharpe':>8} {'RecFactor':>12} {'PF':>6}"
    print(hdr)
    print('-' * 88)
    for name, res in [('Ridge Meta',     res_ridge_oos),
                       ('CatBoost Meta',  res_cat_oos),
                       ('Equal-Weight',   res_ridge_oos)]:
        m = res['meta_metrics'] if 'Meta' in name else res['eq_metrics']
        print(f"{name:<22} {m['net_profit']:>+10,.0f}  "
              f"{m['max_dd_pct']:>7.2f}%  "
              f"{m['sharpe']:>+8.3f}  "
              f"{m['recovery_factor']:>12.2f}x  "
              f"{m['profit_factor']:>6.2f}")
    print('=' * 88)

    # Per-asset breakdown
    print(f'\n{"Ticker":<8} {"Horizon":>8} {"OOS Net($)":>12} {"MaxDD%":>8} {"Sharpe":>8} {"PF":>6}')
    print('-' * 56)
    idx_oos_ref = res_ridge_oos['meta_pnl'].index
    for ticker in ASSETS:
        p    = pnl_all_[ticker].reindex(idx_oos_ref).fillna(0)
        net  = float(p.sum() * INITIAL_CAPITAL)
        sh   = asset_sharpe(p)
        pf   = profit_factor(p)
        cum  = p.cumsum()
        mdd  = float((cum - cum.cummax()).min() * 100)
        print(f"  {ticker:<6} h={ASSET_HORIZONS[ticker]}d  {net:>+12,.0f}  "
              f"{mdd:>7.2f}%  {sh:>+8.3f}  {pf:>6.2f}")

    # ── Step 7: Charts ────────────────────────────────────────────────────────
    print('\n[Step 7] Generating charts')
    plot_results(res_ridge_oos, res_cat_oos,
                 res_ridge_val, res_cat_val,
                 es_sharpe_10d_oos=sh_es_oos)

    # Weights-over-time chart (separate clean chart)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    for ax_w2, res_o, name in [(axes2[0], res_ridge_oos, 'Ridge Meta'),
                                 (axes2[1], res_cat_oos,   'CatBoost Meta')]:
        wt_monthly = res_o['weight_ts'].resample('ME').mean()
        bottom = np.zeros(len(wt_monthly))
        for ticker in ASSETS:
            c = COLORS[ticker]
            vals = wt_monthly[ticker].fillna(EQ_WEIGHT).values
            ax_w2.bar(wt_monthly.index, vals, bottom=bottom,
                      color=c, alpha=0.82, width=20, label=ticker)
            bottom += vals
        ax_w2.axhline(TOTAL_LEVERAGE, color='k', lw=1.2, ls='--',
                      label=f'Total leverage ({TOTAL_LEVERAGE}x)')
        ax_w2.set_ylim(0, TOTAL_LEVERAGE + 1)
        ax_w2.set_title(f'{name} — OOS Leverage Allocation over Time', fontsize=10)
        ax_w2.set_ylabel('Total Leverage (stacked)', fontsize=9)
        ax_w2.legend(fontsize=8, loc='lower right', ncol=3)
        ax_w2.grid(axis='y', alpha=0.25)
        ax_w2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}x'))

    plt.tight_layout()
    wt_path = os.path.join(META_DIR, 'meta_weights_over_time.png')
    plt.savefig(wt_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {wt_path}')

    # Clean equity-only chart
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    for (res_v, res_o, name, color) in [
        (res_ridge_val, res_ridge_oos, 'Ridge Meta', '#27AE60'),
        (res_cat_val,   res_cat_oos,   'CatBoost Meta', '#8E44AD'),
    ]:
        off = res_v['meta_equity'].iloc[-1] - INITIAL_CAPITAL
        (res_v['meta_equity'] / 1000).plot(ax=ax3, color=color, lw=2.2,
                                            label=f'{name} VAL ({res_v["meta_metrics"]["net_profit"]:+,.0f}$)')
        ((res_o['meta_equity'] + off) / 1000).plot(
            ax=ax3, color=color, lw=2.2, ls='--',
            label=f'{name} OOS ({res_o["meta_metrics"]["net_profit"]:+,.0f}$)')

    off_eq = res_ridge_val['eq_equity'].iloc[-1] - INITIAL_CAPITAL
    (res_ridge_val['eq_equity'] / 1000).plot(ax=ax3, color='#2980B9', lw=1.8, ls=':',
                                              label=f'Equal-Weight VAL ({res_ridge_val["eq_metrics"]["net_profit"]:+,.0f}$)')
    ((res_ridge_oos['eq_equity'] + off_eq) / 1000).plot(
        ax=ax3, color='#2980B9', lw=1.8, ls=':',
        label=f'Equal-Weight OOS ({res_ridge_oos["eq_metrics"]["net_profit"]:+,.0f}$)')

    ax3.axvline(pd.Timestamp(OOS_START), color='red', lw=1.5, ls=':', label='OOS start')
    ax3.axhline(INITIAL_CAPITAL / 1000, color='k', lw=0.7)
    ax3.set_ylim(85, None)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}k'))
    ax3.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax3.set_title(
        f'Meta-Arbitrator vs Equal-Weight  |  $100k  |  {TOTAL_LEVERAGE}x leverage  |  '
        f'6 assets (ES 10d restored)\n'
        f'Ridge OOS Sharpe={res_ridge_oos["meta_metrics"]["sharpe"]:+.3f}  |  '
        f'CatBoost OOS Sharpe={res_cat_oos["meta_metrics"]["sharpe"]:+.3f}  |  '
        f'EqW OOS Sharpe={res_ridge_oos["eq_metrics"]["sharpe"]:+.3f}',
        fontsize=10
    )
    ax3.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax3.grid(alpha=0.2)
    plt.tight_layout()
    eq_path = os.path.join(META_DIR, 'meta_equity_clean.png')
    plt.savefig(eq_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {eq_path}')

    elapsed = time.time() - t0
    print(f'\nTotal elapsed: {elapsed:.0f}s')
    print(f'Output: {META_DIR}/')
    print('  meta_arbitrator_report.png')
    print('  meta_weights_over_time.png')
    print('  meta_equity_clean.png')
