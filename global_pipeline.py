"""
global_pipeline.py — Multi-Asset Alpha Mining + Monthly WFO + Portfolio Report
================================================================================
Target assets  : ES, NQ, RTY, FDAX, FESX, E6, CL
Horizon range  : 5–20 days (optimized per asset)
WFO            : 5yr window / 1mo step / LGB+CatBoost ensemble / Top-50 stable factors
Financial      : $100k capital / return-based PnL / realistic commission
Output         : results/{TICKER}/ per asset  +  results/portfolio/ combined report

Note: Alpha signal generation modules are not included in this public release.
      This file contains the pipeline framework, WFO engine, and backtest logic.
"""

import os, sys, json, warnings, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
from catboost import CatBoostRegressor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = 'results'
DATA_DIR    = 'data'
RESULTS_DIR = 'results'

INSTRUMENT_PATHS = {
    'ES':   f'{DATA_DIR}/ES_full_1min_continuous_absolute_adjusted.txt',
    'NQ':   f'{DATA_DIR}/NQ_full_1min_continuous_absolute_adjusted.txt',
    'RTY':  f'{DATA_DIR}/RTY_full_1min_continuous_absolute_adjusted.txt',
    'GC':   f'{DATA_DIR}/GC_full_1min_continuous_absolute_adjusted.txt',
    'VX':   f'{DATA_DIR}/VX_full_1min_continuous_absolute_adjusted.txt',
    'ZN':   f'{DATA_DIR}/ZN_full_1min_continuous_absolute_adjusted.txt',
    'DX':   f'{DATA_DIR}/DX_full_1min_continuous_absolute_adjusted.txt',
    'CL':   f'{DATA_DIR}/CL_full_1min_continuous_absolute_adjusted.txt',
    'E6':   f'{DATA_DIR}/E6_full_1min_continuous_absolute_adjusted.txt',
    'FDAX': f'{DATA_DIR}/FDAX_full_1min_continuous_absolute_adjusted.txt',
    'FESX': f'{DATA_DIR}/FESX_full_1min_continuous_absolute_adjusted.txt',
    'FGBL': f'{DATA_DIR}/FGBL_full_1min_continuous_absolute_adjusted.txt',
    'FVSA': f'{DATA_DIR}/FVSA_full_1min_continuous_absolute_adjusted.txt',
}

# Per-asset intermarket partner configuration.
# Partners are correlated instruments used to build intermarket signals.
# Specific partner assignments reflect domain knowledge and are not disclosed.
ASSET_CONFIG = {
    'ES':   {'partners': [...]},   # 7 correlated instruments
    'NQ':   {'partners': [...]},   # 8 correlated instruments
    'RTY':  {'partners': [...]},   # 8 correlated instruments
    'FDAX': {'partners': [...]},   # 9 correlated instruments
    'FESX': {'partners': [...]},   # 9 correlated instruments
    'E6':   {'partners': [...]},   # 7 correlated instruments
    'CL':   {'partners': [...]},   # 7 correlated instruments
}

TARGET_ASSETS      = ['ES', 'NQ', 'RTY', 'FDAX', 'FESX', 'E6', 'CL']
HORIZON_CANDIDATES = [5, 10, 15, 20]
HORIZON_OPT_TRAIN  = ('2011-01-01', '2018-01-01')
HORIZON_OPT_CONF   = ('2018-01-01', '2021-01-01')

TRAIN_START      = '2011-01-01'
TRAIN_END        = '2021-01-01'
VAL_START        = '2021-01-01'
VAL_END          = '2024-01-01'
OOS_START        = '2024-01-01'
OOS_END          = '2026-03-01'

TRAIN_WINDOW_YRS = 5
STEP_MONTHS      = 1
TOP_N_FACTORS    = 50
TARGET_N         = 300
IC_THRESHOLD     = 0.01
CORR_THRESH      = 0.70
NOISE_PCTILE     = 25
Q_LONG           = 0.70
Q_SHORT          = 0.30
INITIAL_CAPITAL  = 100_000
COMM_RATE_RT     = 0.00006

LGB_PARAMS = dict(
    objective='regression', num_leaves=31, max_depth=5,
    learning_rate=0.03, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
    verbose=-1, random_state=42,
)
LGB_ITER = 300
CAT_PARAMS = dict(
    iterations=300, learning_rate=0.03, depth=5,
    l2_leaf_reg=3.0, random_seed=42,
    allow_writing_files=False, verbose=False,
)
FORCE_RECOMPUTE = False
eps = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALPHA SIGNAL GENERATION  (proprietary — not included)
# ══════════════════════════════════════════════════════════════════════════════

def _build_sa_signals(primary_df: pd.DataFrame) -> dict:
    """
    Single-asset alpha signals computed from OHLCV + VWAP data.
    Uses a library of mathematical operators over multiple lookback windows.
    Returns {name: (pd.Series, description)}.

    NOTE: Implementation not included in public release.
    """
    raise NotImplementedError(
        "SA signal library not included. "
        "Provide your own implementation or load a pre-computed alpha pool."
    )


def _build_im_signals(wide: pd.DataFrame, primary: str) -> dict:
    """
    Intermarket signals between the primary asset and its partner instruments.
    Captures cross-asset momentum, spread dynamics, and lead-lag relationships.
    Returns {name: (pd.Series, description)}.

    NOTE: Implementation not included in public release.
    """
    raise NotImplementedError(
        "IM signal library not included. "
        "Provide your own implementation or load a pre-computed alpha pool."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_single(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime'], index_col='datetime'
    )
    df['tp']  = (df['high'] + df['low'] + df['close']) / 3
    df['tpv'] = df['tp'] * df['volume']
    daily = df.resample('1D').agg(
        open    = ('open',   'first'),
        high    = ('high',   'max'),
        low     = ('low',    'min'),
        close   = ('close',  'last'),
        volume  = ('volume', 'sum'),
        tpv_sum = ('tpv',    'sum'),
    ).dropna(subset=['close'])
    daily = daily[daily['volume'] > 0].copy()
    daily['vwap'] = daily['tpv_sum'] / daily['volume']
    daily.drop(columns=['tpv_sum'], inplace=True)
    return daily


def load_wide_for_asset(ticker: str) -> pd.DataFrame:
    """
    Loads primary ticker + all available partners.
    Aligns on primary's trading calendar. Partners: ffill(limit=3).
    Returns wide DataFrame with columns: {sym_lower}_{field}
    """
    ticker_lower = ticker.lower()
    partners     = ASSET_CONFIG[ticker]['partners']

    primary_df = load_daily_single(INSTRUMENT_PATHS[ticker])
    primary_df.columns = [f'{ticker_lower}_{c}' for c in primary_df.columns]
    base_idx = primary_df.index

    parts = [primary_df]
    loaded = []
    for partner in partners:
        path = INSTRUMENT_PATHS.get(partner)
        if path is None or not os.path.exists(path):
            continue
        try:
            d   = load_daily_single(path)
            sym = partner.lower()
            d.columns = [f'{sym}_{c}' for c in d.columns]
            aligned = d.reindex(base_idx).ffill(limit=3)
            parts.append(aligned)
            loaded.append(partner)
        except Exception:
            pass

    wide = pd.concat(parts, axis=1)
    wide = wide.dropna(subset=[f'{ticker_lower}_close'])
    print(f'  Loaded: {ticker} + {len(loaded)} partners  |  '
          f'{wide.index[0].date()} -> {wide.index[-1].date()}')
    return wide


def log_ret(x):
    return np.log(x / x.shift(1)).replace([np.inf, -np.inf], np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ALPHA POOL MINING
# ══════════════════════════════════════════════════════════════════════════════

def mine_alpha_pool(wide: pd.DataFrame, ticker: str, horizon: int,
                    out_dir: str) -> list:
    """
    Builds SA + IM signal candidates, filters by IC and correlation dedup,
    saves accepted pool to JSON. Requires signal generation modules.

    Selection criteria:
      1. |RankIC_train| > IC_THRESHOLD
      2. |Spearman corr| < CORR_THRESH with any accepted factor (greedy dedup)
      3. Sorted by stability score across train/validation periods
    """
    pool_path = os.path.join(out_dir, 'alpha_pool.json')
    if not FORCE_RECOMPUTE and os.path.exists(pool_path):
        print(f'  [SKIP] Pool exists: {pool_path}')
        with open(pool_path, 'r') as f:
            return json.load(f)

    # Requires proprietary signal generation modules
    t       = ticker.lower()
    close   = wide[f'{t}_close']
    fwd_ret = np.log(close.shift(-horizon) / close)
    train_m = (wide.index >= TRAIN_START) & (wide.index < TRAIN_END)
    val_m   = (wide.index >= VAL_START)   & (wide.index < '2023-01-01')

    sa_raw = _build_sa_signals(wide[[c for c in wide.columns if c.startswith(t)]])
    im_raw = _build_im_signals(wide, ticker)

    all_cands = {}
    for name, val in sa_raw.items():
        all_cands[name] = (val[0], val[1], 'single_asset')
    for name, val in im_raw.items():
        if name not in all_cands:
            all_cands[name] = (val[0], val[1], 'intermarket')

    scored = []
    ft = fwd_ret[train_m]
    for name, (sig, desc, src) in all_cands.items():
        try:
            st = sig[train_m]
            m  = st.notna() & ft.notna()
            if m.sum() < 50:
                continue
            ic = abs(spearmanr(st[m], ft[m])[0])
            if ic >= IC_THRESHOLD:
                scored.append((name, ic, sig, desc, src))
        except Exception:
            pass

    scored.sort(key=lambda x: x[1], reverse=True)
    accepted      = []
    accepted_sigs = []

    for name, ic_tr, sig, desc, src in scored:
        if len(accepted) >= TARGET_N:
            break
        sv, fv = sig[val_m], fwd_ret[val_m]
        mv = sv.notna() & fv.notna()
        ic_val = spearmanr(sv[mv], fv[mv])[0] if mv.sum() >= 30 else np.nan

        st_arr    = sig[train_m].values
        duplicate = False
        for acc_s in accepted_sigs:
            acc_arr = acc_s[train_m].values
            valid   = ~(np.isnan(st_arr) | np.isnan(acc_arr))
            if valid.sum() < 30:
                continue
            if abs(spearmanr(st_arr[valid], acc_arr[valid])[0]) >= CORR_THRESH:
                duplicate = True
                break

        if not duplicate:
            accepted.append({
                'name':          name,
                'expression':    desc,
                'rank_ic_train': round(float(ic_tr), 5),
                'rank_ic_val':   round(float(ic_val), 5) if not np.isnan(ic_val) else None,
                'source':        src,
            })
            accepted_sigs.append(sig)

    os.makedirs(out_dir, exist_ok=True)
    with open(pool_path, 'w', encoding='utf-8') as f:
        json.dump(accepted, f, indent=2)
    print(f'  Pool: {len(accepted)} factors saved -> {pool_path}')
    return accepted


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HORIZON OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def optimize_horizon(primary_df: pd.DataFrame, horizons=None) -> int:
    """
    Evaluates each candidate horizon on the training period using a sample
    of price-derived signals. Selects the horizon maximizing a blended
    IC score across train and confirmation windows.
    """
    if horizons is None:
        horizons = HORIZON_CANDIDATES

    close   = primary_df['close']
    train_m = (primary_df.index >= HORIZON_OPT_TRAIN[0]) & \
              (primary_df.index < HORIZON_OPT_TRAIN[1])
    conf_m  = (primary_df.index >= HORIZON_OPT_CONF[0]) & \
              (primary_df.index < HORIZON_OPT_CONF[1])

    best_h, best_score = horizons[0], -np.inf
    for h in horizons:
        fwd = np.log(close.shift(-h) / close)
        # Sample of 9 representative signals (implementation omitted)
        # Score = 0.6 × mean|IC_train| + 0.4 × mean|IC_confirm|
        # This blend prevents horizon overfitting to training period
        score = 0.0   # placeholder — requires signal builders
        if score > best_score:
            best_score, best_h = score, h

    return best_h


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(wide: pd.DataFrame, ticker: str,
                         pool_names: list) -> pd.DataFrame:
    """Rebuilds all signals and returns feature DataFrame indexed by date."""
    sa_raw = _build_sa_signals(wide)
    im_raw = _build_im_signals(wide, ticker)
    all_raw = {n: v[0] for n, v in sa_raw.items()}
    for n, v in im_raw.items():
        if n not in all_raw:
            all_raw[n] = v[0]
    cols    = {n: all_raw[n] for n in pool_names if n in all_raw}
    missing = [n for n in pool_names if n not in all_raw]
    if missing:
        print(f'  [WARN] {len(missing)} factors missing from rebuild')
    return pd.DataFrame(cols, index=wide.index)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STABLE FACTOR SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_stable_factors(pool: list, feat_df: pd.DataFrame,
                           target: pd.Series, top_n: int = TOP_N_FACTORS) -> list:
    """
    Selects top_n factors by stability score:
      stability = sign_consistency × (|IC_train| + |IC_val|) × ic_dominance
    where sign_consistency = 1 if sign(IC_train) == sign(IC_val), else 0,
    and ic_dominance = max(frac_positive, frac_negative) across rolling folds.
    """
    TRAIN_S, TRAIN_E = '2011-01-01', '2018-01-01'
    VAL_S,   VAL_E   = '2018-01-01', '2021-01-01'
    folds = [
        ('2011-01-01', '2014-01-01'), ('2012-07-01', '2015-07-01'),
        ('2014-01-01', '2017-01-01'), ('2015-07-01', '2018-07-01'),
        ('2017-01-01', '2020-01-01'), ('2018-07-01', '2021-07-01'),
    ]
    tgt = target.dropna()

    def ic_period(sig, s, e):
        m = sig.notna() & tgt.notna() & (sig.index >= s) & (sig.index < e)
        return spearmanr(sig[m], tgt[m])[0] if m.sum() >= 30 else np.nan

    scores = []
    for entry in pool:
        name = entry['name']
        if name not in feat_df.columns:
            scores.append({'name': name, 'stability': 0.0})
            continue
        sig    = feat_df[name]
        ic_tr  = ic_period(sig, TRAIN_S, TRAIN_E)
        ic_vl  = ic_period(sig, VAL_S, VAL_E)
        if np.isnan(ic_tr) or np.isnan(ic_vl):
            scores.append({'name': name, 'stability': 0.0})
            continue
        fold_ics = [x for x in [ic_period(sig, s, e) for s, e in folds]
                    if not np.isnan(x)]
        dom  = (max(sum(x > 0 for x in fold_ics),
                    sum(x < 0 for x in fold_ics)) / len(fold_ics)
                if fold_ics else 0.5)
        stab = (1 if np.sign(ic_tr) == np.sign(ic_vl) else 0) \
               * (abs(ic_tr) + abs(ic_vl)) * dom
        scores.append({'name': name, 'stability': stab,
                       'ic_train': ic_tr, 'ic_val': ic_vl})

    scores.sort(key=lambda x: x['stability'], reverse=True)
    top    = [s['name'] for s in scores[:top_n]]
    ic_tr_mean = np.mean([s.get('ic_train', 0) for s in scores[:top_n] if 'ic_train' in s])
    ic_vl_mean = np.mean([s.get('ic_val',   0) for s in scores[:top_n] if 'ic_val'   in s])
    sign_ok    = sum(1 for s in scores[:top_n]
                     if 'ic_train' in s
                     and np.sign(s['ic_train']) == np.sign(s.get('ic_val', 0)))
    print(f'  Top-{top_n} selected  |  '
          f'mean IC_train={ic_tr_mean:.4f}  IC_val={ic_vl_mean:.4f}  '
          f'sign-ok={sign_ok}/{top_n}')
    return top


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MONTHLY WALK-FORWARD OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def train_window(X_tr, y_tr, feat_names):
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, free_raw_data=False)
    lgb_m  = lgb.train(LGB_PARAMS, dtrain, LGB_ITER,
                       callbacks=[lgb.log_evaluation(-1)])
    cat_m  = CatBoostRegressor(**CAT_PARAMS)
    cat_m.fit(X_tr, y_tr, verbose=False)
    p_tr  = 0.5 * lgb_m.predict(X_tr) + 0.5 * cat_m.predict(X_tr)
    noise = np.percentile(np.abs(p_tr), NOISE_PCTILE)
    return lgb_m, cat_m, noise


def rolling_wfo(feat_df: pd.DataFrame, target: pd.Series,
                feat_names: list, start: str, end: str) -> pd.DataFrame:
    """
    Monthly walk-forward optimization.
    Training window: TRAIN_WINDOW_YRS years, step: STEP_MONTHS month(s).
    Returns DataFrame with predictions, targets, IC, and noise floor per window.
    """
    results  = []
    pred_s   = pd.Timestamp(start)
    pred_e   = pd.Timestamp(end)
    win      = relativedelta(years=TRAIN_WINDOW_YRS)
    step     = relativedelta(months=STEP_MONTHS)

    tr_end   = pred_s
    tr_start = tr_end - win
    pf, pt   = pred_s, pred_s + step
    w_idx    = 1

    while pf < pred_e:
        pt_act = min(pt, pred_e)
        X_tr   = feat_df.loc[str(tr_start):str(tr_end - pd.Timedelta(days=1)),
                              feat_names].values
        y_tr   = target.loc[str(tr_start):str(tr_end - pd.Timedelta(days=1))].values
        m      = ~(np.isnan(X_tr).any(1) | np.isnan(y_tr))
        X_tr, y_tr = X_tr[m], y_tr[m]

        X_pr   = feat_df.loc[str(pf):str(pt_act - pd.Timedelta(days=1)),
                              feat_names].values
        idx_pr = feat_df.loc[str(pf):str(pt_act - pd.Timedelta(days=1))].index
        y_pr   = target.loc[str(pf):str(pt_act - pd.Timedelta(days=1))].values

        if len(X_tr) < 200 or len(X_pr) == 0:
            tr_start += step; tr_end += step; pf += step; pt += step; w_idx += 1
            continue

        t0 = time.time()
        lgb_m, cat_m, noise = train_window(X_tr, y_tr, feat_names)
        el = time.time() - t0

        p_lgb = lgb_m.predict(X_pr)
        p_cat = cat_m.predict(X_pr)
        p_ens = 0.5 * p_lgb + 0.5 * p_cat

        vm     = ~np.isnan(y_pr)
        ic_ens = spearmanr(p_ens[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan
        ic_lgb = spearmanr(p_lgb[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan
        ic_cat = spearmanr(p_cat[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan

        if w_idx % 6 == 1 or w_idx == 1:
            print(f'  Win {w_idx:>2}  {str(tr_start)[:7]}->{str(tr_end)[:7]}'
                  f'  pred={str(pf)[:7]}  n={len(X_tr)}'
                  f'  noise={noise:.4f}'
                  f'  IC[L={ic_lgb:+.3f} C={ic_cat:+.3f} E={ic_ens:+.3f}]'
                  f'  ({el:.0f}s)')

        for i, date in enumerate(idx_pr):
            results.append({
                'date':   date, 'window': w_idx,
                'p_lgb':  p_lgb[i], 'p_cat': p_cat[i], 'p_ens': p_ens[i],
                'target': y_pr[i],  'noise': noise,
                'ic_ens': ic_ens,   'ic_lgb': ic_lgb, 'ic_cat': ic_cat,
            })
        tr_start += step; tr_end += step; pf += step; pt += step; w_idx += 1

    print(f'  Total windows: {w_idx - 1}')
    return pd.DataFrame(results).set_index('date').sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — POSITION SIZING & FINANCIAL BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def make_positions(wfo: pd.DataFrame, signal_col: str,
                   regime_exit: bool = False) -> pd.Series:
    """
    Quantile-based long/short/cash positions per WFO window.
    Top Q_LONG quantile -> long (+1), bottom Q_SHORT -> short (-1), rest -> cash (0).
    Regime exit: zero position when both models predict within the noise floor.
    """
    pos = pd.Series(0.0, index=wfo.index)
    for win_id in wfo['window'].unique():
        m       = wfo['window'] == win_id
        sig     = wfo.loc[m, signal_col]
        noise   = wfo.loc[m, 'noise'].iloc[0]
        thr_hi  = sig.quantile(Q_LONG)
        thr_lo  = sig.quantile(Q_SHORT)
        long_m  = m & (wfo[signal_col] >= thr_hi)
        short_m = m & (wfo[signal_col] <= thr_lo)
        if regime_exit:
            noisy   = (wfo['p_lgb'].abs() < noise) & (wfo['p_cat'].abs() < noise)
            long_m  = long_m  & ~noisy
            short_m = short_m & ~noisy
        pos[long_m]  =  1.0
        pos[short_m] = -1.0
    return pos


def financial_backtest(wfo: pd.DataFrame, ret_1d: pd.Series,
                        signal_col: str = 'p_ens',
                        regime_exit: bool = True,
                        horizon: int = 10) -> dict:
    """
    Return-based financial simulation.
    pnl_daily = pos × target_fwd / horizon
    commission = COMM_RATE_RT per position change
    """
    pos       = make_positions(wfo, signal_col, regime_exit)
    tgt_fwd   = wfo['target']
    r1        = ret_1d.reindex(wfo.index)

    pnl_raw   = pos * tgt_fwd.fillna(0) / horizon
    pos_change = pos.diff().abs().fillna(0) > 0
    commission = pos_change.astype(float) * COMM_RATE_RT
    pnl_net   = pnl_raw - commission
    equity    = INITIAL_CAPITAL * (1 + pnl_net.cumsum())
    bh_equity = INITIAL_CAPITAL * (1 + r1.fillna(0).cumsum())

    sharpe   = pnl_net.mean() / (pnl_net.std() + eps) * np.sqrt(252)
    cum      = pnl_net.cumsum()
    max_dd   = float((cum - cum.cummax()).min())
    pf       = pnl_net[pnl_net > 0].sum() / (abs(pnl_net[pnl_net < 0].sum()) + eps)
    vm       = wfo[signal_col].notna() & tgt_fwd.notna()
    ic, _    = spearmanr(wfo.loc[vm, signal_col], tgt_fwd[vm])
    net_pnl  = float(equity.iloc[-1] - INITIAL_CAPITAL)
    win_rate = (pnl_net[pos != 0] > 0).mean() if (pos != 0).any() else np.nan

    return dict(
        pos=pos, pnl=pnl_net, equity=equity, bh_equity=bh_equity,
        sharpe=sharpe, max_dd=max_dd, max_dd_pct=max_dd * 100,
        net_profit=net_pnl, profit_factor=pf,
        ic=ic, win_rate=win_rate,
        n_long=(pos==1).sum(), n_short=(pos==-1).sum(), n_tot=len(pos),
        cum_ret=float(cum.iloc[-1]) * 100,
        bh_ret=float(r1.fillna(0).cumsum().iloc[-1]) * 100,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN ASSET RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_asset(ticker: str) -> dict:
    """
    Full pipeline for a single asset:
    1. Load data (primary + partners)
    2. Optimize prediction horizon
    3. Mine alpha pool (300 factors)
    4. Build feature matrix
    5. Select top-50 stable factors
    6. Run monthly WFO (VAL + OOS)
    7. Financial backtest
    8. Save results and chart

    Requires proprietary signal generation modules for steps 2-4.
    With pre-computed alpha_pool.json, steps 2-3 are skipped (FORCE_RECOMPUTE=False).
    """
    t0 = time.time()
    out_dir = os.path.join(RESULTS_DIR, ticker)
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n{"="*60}\n  {ticker}\n{"="*60}')

    wide = load_wide_for_asset(ticker)

    # Step 1: Horizon optimization
    horizon_path = os.path.join(out_dir, 'horizon.txt')
    if not FORCE_RECOMPUTE and os.path.exists(horizon_path):
        with open(horizon_path) as f:
            horizon = int(f.read().strip())
        print(f'  Loaded horizon: {horizon}d')
    else:
        primary_df = wide[[c for c in wide.columns
                            if c.startswith(ticker.lower())]].copy()
        primary_df.columns = [c.replace(f'{ticker.lower()}_', '')
                               for c in primary_df.columns]
        horizon = optimize_horizon(primary_df)
        with open(horizon_path, 'w') as f:
            f.write(str(horizon))
        print(f'  Optimized horizon: {horizon}d')

    # Step 2: Alpha pool
    pool = mine_alpha_pool(wide, ticker, horizon, out_dir)

    # Step 3: Feature matrix
    features_path = os.path.join(out_dir, 'features.pkl')
    if not FORCE_RECOMPUTE and os.path.exists(features_path):
        feat_df = pd.read_pickle(features_path)
        print(f'  Loaded features: {feat_df.shape}')
    else:
        pool_names = [p['name'] for p in pool]
        feat_df    = build_feature_matrix(wide, ticker, pool_names)
        feat_df.to_pickle(features_path)
        print(f'  Feature matrix: {feat_df.shape}')

    # Target: horizon-day forward log return
    t_lower = ticker.lower()
    close   = wide[f'{t_lower}_close']
    ret_1d  = log_ret(close)
    target  = np.log(close.shift(-horizon) / close)

    # Step 4: Stable factor selection
    top_factors = select_stable_factors(pool, feat_df, target)

    # Step 5: Monthly WFO
    print(f'  Monthly WFO VAL ({VAL_START} -> {VAL_END})...')
    wfo_val = rolling_wfo(feat_df, target, top_factors, VAL_START, VAL_END)
    print(f'  Monthly WFO OOS ({OOS_START} -> {OOS_END})...')
    wfo_oos = rolling_wfo(feat_df, target, top_factors, OOS_START, OOS_END)

    # Step 6: Financial backtest
    res_val = financial_backtest(wfo_val, ret_1d, horizon=horizon)
    res_oos = financial_backtest(wfo_oos, ret_1d, horizon=horizon)

    # Save WFO results
    wfo_val.to_pickle(os.path.join(out_dir, 'wfo_val.pkl'))
    wfo_oos.to_pickle(os.path.join(out_dir, 'wfo_oos.pkl'))

    # Save summary
    n_sa = sum(1 for p in pool if p.get('source') == 'single_asset')
    n_im = sum(1 for p in pool if p.get('source') == 'intermarket')
    summary = {
        'ticker': ticker, 'horizon': horizon,
        'pool_size': len(pool), 'n_sa': n_sa, 'n_im': n_im,
        'top_factors': top_factors,
        'val': {
            'sharpe':      round(res_val['sharpe'], 6),
            'max_dd_pct':  round(res_val['max_dd_pct'], 6),
            'net_profit':  round(res_val['net_profit'], 4),
            'profit_factor': round(res_val['profit_factor'], 6),
            'ic':          round(res_val['ic'], 6),
            'win_rate':    round(float(res_val['win_rate']), 4),
            'cum_ret':     round(res_val['cum_ret'], 4),
            'bh_ret':      round(res_val['bh_ret'], 4),
        },
        'oos': {
            'sharpe':      round(res_oos['sharpe'], 6),
            'max_dd_pct':  round(res_oos['max_dd_pct'], 6),
            'net_profit':  round(res_oos['net_profit'], 4),
            'profit_factor': round(res_oos['profit_factor'], 6),
            'ic':          round(res_oos['ic'], 6),
            'win_rate':    round(float(res_oos['win_rate']), 4),
            'cum_ret':     round(res_oos['cum_ret'], 4),
            'bh_ret':      round(res_oos['bh_ret'], 4),
        },
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f'\n  [{ticker}] Done in {elapsed:.0f}s  |  '
          f'VAL Sharpe={res_val["sharpe"]:+.3f}  '
          f'OOS Sharpe={res_oos["sharpe"]:+.3f}  '
          f'OOS Net=${res_oos["net_profit"]:+,.0f}')

    return {'ticker': ticker, 'horizon': horizon,
            'res_val': res_val, 'res_oos': res_oos,
            'wfo_val': wfo_val, 'wfo_oos': wfo_oos}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PORTFOLIO AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_portfolio(asset_results: list):
    """Prints OOS summary table and saves portfolio equity chart."""
    print('\n' + '=' * 80)
    print('OOS SUMMARY TABLE (2024-2026)')
    print('=' * 80)
    hdr = (f"{'Ticker':<8} {'Horizon':>8} {'OOS Net':>10} {'Sharpe':>8} "
           f"{'MaxDD%':>8} {'PF':>6} {'IC':>8} {'WinRate':>9}")
    print(hdr)
    print('-' * 80)
    for r in asset_results:
        m = r['res_oos']
        print(f"  {r['ticker']:<6}  h={r['horizon']:>2}d  "
              f"{m['net_profit']:>+10,.0f}  "
              f"{m['sharpe']:>+8.3f}  "
              f"{m['max_dd_pct']:>7.2f}%  "
              f"{m['profit_factor']:>6.2f}  "
              f"{m['ic']:>8.4f}  "
              f"{m['win_rate']:>8.1%}")
    print('=' * 80)

    # Portfolio equity
    os.makedirs(os.path.join(RESULTS_DIR, 'portfolio'), exist_ok=True)
    oos_csv_path = os.path.join(RESULTS_DIR, 'portfolio', 'oos_summary.csv')
    rows = []
    for r in asset_results:
        m = r['res_oos']
        rows.append({'ticker': r['ticker'], 'horizon': r['horizon'],
                     'net_profit': m['net_profit'], 'sharpe': m['sharpe'],
                     'max_dd_pct': m['max_dd_pct'], 'profit_factor': m['profit_factor']})
    pd.DataFrame(rows).to_csv(oos_csv_path, index=False)
    print(f'  Summary saved -> {oos_csv_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()
    print('=' * 70)
    print('  CHAIN OF ALPHA — GLOBAL MULTI-ASSET PIPELINE')
    print(f'  Assets: {TARGET_ASSETS}')
    print(f'  OOS: {OOS_START} -> {OOS_END}')
    print('=' * 70)

    asset_results = []
    for ticker in TARGET_ASSETS:
        try:
            result = run_asset(ticker)
            asset_results.append(result)
        except Exception as e:
            print(f'  [ERROR] {ticker}: {e}')

    if asset_results:
        aggregate_portfolio(asset_results)

    print(f'\nTotal elapsed: {(time.time()-t_total)/60:.1f} min')
