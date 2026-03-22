"""
Monthly WFO — LightGBM vs CatBoost vs Ensemble + Regime Exit
==============================================================
Изменения vs rolling_wfo.py:
  1. Шаг переобучения: 1 месяц (было 6)
  2. Независимые модели LGB / CatBoost + сравнительная таблица
  3. Ensemble 2.0 = среднее прогнозов LGB и CatBoost
  4. Regime Exit: если обе модели в «зоне шума» → позиция = 0
  5. Сравнительный вывод Sharpe / MaxDD / IC / CumRet по VAL и OOS
"""

import json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from generate_alpha_pool import build_all_signals
from generate_intermarket import load_all_aligned, build_intermarket_signals, log_ret

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
POOL_FILE        = 'alpha_pool_final.json'
TRAIN_WINDOW_YRS = 5
STEP_MONTHS      = 1          # ← месячный шаг
TOP_N_FACTORS    = 50
FORWARD_DAYS     = 10
Q_LONG           = 0.70       # топ 30% → лонг
Q_SHORT          = 0.30       # боттом 30% → шорт
NOISE_PCTILE     = 25         # p25 |pred_train| → порог шума (Regime Exit)

WFO_START = '2021-01-01'
WFO_END   = '2024-01-01'
OOS_START = '2024-01-01'
OOS_END   = '2026-03-01'

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'num_leaves': 15, 'learning_rate': 0.02,
    'feature_fraction': 0.50, 'bagging_fraction': 0.70, 'bagging_freq': 1,
    'min_data_in_leaf': 50, 'lambda_l1': 1.0, 'lambda_l2': 1.0,
    'max_depth': 4, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}
LGB_ITER = 300

CAT_PARAMS = dict(
    iterations=300, learning_rate=0.02, depth=4, l2_leaf_reg=10.0,
    rsm=0.5, subsample=0.70, random_seed=42,
    allow_writing_files=False, verbose=False,
)
eps = 1e-8


# ─────────────────────────────────────────────
# 1. FEATURE MATRIX
# ─────────────────────────────────────────────
def build_feature_matrix(full, pool_names):
    d_es = full[['es_open','es_high','es_low','es_close','es_volume','es_vwap']].copy()
    d_es.columns = ['open','high','low','close','volume','vwap']
    sa  = {n: v[0] for n, v in build_all_signals(d_es).items()}
    im  = {n: v[0] for n, v in build_intermarket_signals(full).items() if n not in sa}
    all_sigs = {**sa, **im}
    cols = {n: all_sigs[n] for n in pool_names if n in all_sigs}
    return pd.DataFrame(cols, index=full.index)


# ─────────────────────────────────────────────
# 2. FACTOR STABILITY
# ─────────────────────────────────────────────
def select_stable_factors(pool, feat_df, target, top_n=TOP_N_FACTORS):
    TRAIN_S, TRAIN_E = '2011-01-01', '2018-01-01'
    VAL_S,   VAL_E   = '2018-01-01', '2021-01-01'
    folds = [('2011-01-01','2014-01-01'),('2012-07-01','2015-07-01'),
             ('2014-01-01','2017-01-01'),('2015-07-01','2018-07-01'),
             ('2017-01-01','2020-01-01'),('2018-07-01','2021-07-01')]
    tgt = target.dropna()
    scores = []
    for entry in pool:
        name = entry['name']
        if name not in feat_df.columns:
            scores.append({'name': name, 'stability': 0.0, 'ic_train': 0.0, 'ic_val': 0.0}); continue
        sig = feat_df[name]
        def ic_p(s, e):
            m = sig.notna() & tgt.notna() & (sig.index >= s) & (sig.index < e)
            return spearmanr(sig[m], tgt[m])[0] if m.sum() >= 30 else np.nan
        ic_tr, ic_vl = ic_p(TRAIN_S, TRAIN_E), ic_p(VAL_S, VAL_E)
        if np.isnan(ic_tr) or np.isnan(ic_vl):
            scores.append({'name': name, 'stability': 0.0, 'ic_train': 0.0, 'ic_val': 0.0}); continue
        fold_ics = [x for x in [ic_p(s,e) for s,e in folds] if not np.isnan(x)]
        dom = max(sum(x>0 for x in fold_ics), sum(x<0 for x in fold_ics)) / len(fold_ics) if fold_ics else 0.5
        stab = (1 if np.sign(ic_tr)==np.sign(ic_vl) else 0) * (abs(ic_tr)+abs(ic_vl)) * dom
        scores.append({'name': name, 'stability': stab, 'ic_train': ic_tr, 'ic_val': ic_vl})
    df = pd.DataFrame(scores).sort_values('stability', ascending=False)
    top = df.head(top_n)
    print(f'  Top-{top_n} selected  |  mean IC_train={top.ic_train.abs().mean():.4f}'
          f'  IC_val={top.ic_val.abs().mean():.4f}  sign-ok={int((top.ic_train*top.ic_val>0).sum())}/{top_n}')
    return top['name'].tolist()


# ─────────────────────────────────────────────
# 3. TRAIN INDEPENDENT MODELS
# ─────────────────────────────────────────────
def train_window(X_tr, y_tr, feat_names):
    """Returns lgb_model, cat_model, noise_floor."""
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, free_raw_data=False)
    lgb_m  = lgb.train(LGB_PARAMS, dtrain, LGB_ITER, callbacks=[lgb.log_evaluation(-1)])

    cat_m = CatBoostRegressor(**CAT_PARAMS)
    cat_m.fit(X_tr, y_tr, verbose=False)

    # Noise floor: p25 of |ensemble pred| on training set
    p_tr  = 0.5*lgb_m.predict(X_tr) + 0.5*cat_m.predict(X_tr)
    noise = np.percentile(np.abs(p_tr), NOISE_PCTILE)
    return lgb_m, cat_m, noise


# ─────────────────────────────────────────────
# 4. ROLLING WFO (monthly)
# ─────────────────────────────────────────────
def rolling_wfo(feat_df, target, feat_names, start, end, label=''):
    results = []
    pred_s  = pd.Timestamp(start)
    pred_e  = pd.Timestamp(end)
    win     = relativedelta(years=TRAIN_WINDOW_YRS)
    step    = relativedelta(months=STEP_MONTHS)

    tr_end   = pred_s
    tr_start = tr_end - win
    pf       = pred_s
    pt       = pf + step
    w_idx    = 1

    while pf < pred_e:
        pt_act = min(pt, pred_e)
        X_tr = feat_df.loc[str(tr_start):str(tr_end - pd.Timedelta(days=1)), feat_names].values
        y_tr = target.loc[str(tr_start):str(tr_end - pd.Timedelta(days=1))].values
        m    = ~(np.isnan(X_tr).any(1) | np.isnan(y_tr))
        X_tr, y_tr = X_tr[m], y_tr[m]

        X_pr = feat_df.loc[str(pf):str(pt_act - pd.Timedelta(days=1)), feat_names].values
        idx_pr = feat_df.loc[str(pf):str(pt_act - pd.Timedelta(days=1))].index
        y_pr   = target.loc[str(pf):str(pt_act - pd.Timedelta(days=1))].values

        if len(X_tr) < 200 or len(X_pr) == 0:
            tr_start += step; tr_end += step; pf += step; pt += step; w_idx += 1; continue

        t0 = time.time()
        lgb_m, cat_m, noise = train_window(X_tr, y_tr, feat_names)
        el = time.time() - t0

        p_lgb = lgb_m.predict(X_pr)
        p_cat = cat_m.predict(X_pr)
        p_ens = 0.5*p_lgb + 0.5*p_cat

        vm  = ~np.isnan(y_pr)
        ic_ens = spearmanr(p_ens[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan
        ic_lgb = spearmanr(p_lgb[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan
        ic_cat = spearmanr(p_cat[vm], y_pr[vm])[0] if vm.sum() >= 10 else np.nan

        if w_idx % 6 == 1 or w_idx == 1:   # print every 6 months
            print(f'  Win {w_idx:>2}  {str(tr_start)[:7]}->{str(tr_end)[:7]}'
                  f'  pred {str(pf)[:7]}  n={len(X_tr)}'
                  f'  noise={noise:.4f}  IC[L={ic_lgb:+.3f} C={ic_cat:+.3f} E={ic_ens:+.3f}]'
                  f'  ({el:.0f}s)')

        for i, date in enumerate(idx_pr):
            results.append({
                'date':   date,   'window': w_idx,
                'p_lgb':  p_lgb[i], 'p_cat': p_cat[i], 'p_ens': p_ens[i],
                'target': y_pr[i],  'noise': noise,
                'ic_ens': ic_ens,   'ic_lgb': ic_lgb, 'ic_cat': ic_cat,
            })
        tr_start += step; tr_end += step; pf += step; pt += step; w_idx += 1

    print(f'  Total windows: {w_idx-1}')
    return pd.DataFrame(results).set_index('date').sort_index()


# ─────────────────────────────────────────────
# 5. BACKTEST ENGINE
# ─────────────────────────────────────────────
def make_positions(wfo: pd.DataFrame, signal_col: str,
                   regime_exit: bool = False) -> pd.Series:
    """
    Quantile-based positions computed within each prediction window.
    signal_col: 'p_lgb' | 'p_cat' | 'p_ens'
    regime_exit: if True, apply noise filter (Regime Exit)
    """
    pos = pd.Series(0.0, index=wfo.index)
    for win_id in wfo['window'].unique():
        m    = wfo['window'] == win_id
        sig  = wfo.loc[m, signal_col]
        noise = wfo.loc[m, 'noise'].iloc[0]

        thr_hi = sig.quantile(Q_LONG)
        thr_lo = sig.quantile(Q_SHORT)

        long_m  = m & (wfo[signal_col] >= thr_hi)
        short_m = m & (wfo[signal_col] <= thr_lo)

        if regime_exit:
            # Обе модели должны быть за пределами зоны шума
            noisy = (wfo['p_lgb'].abs() < noise) & (wfo['p_cat'].abs() < noise)
            long_m  = long_m  & ~noisy
            short_m = short_m & ~noisy

        pos[long_m]  =  1.0
        pos[short_m] = -1.0
    return pos


def backtest(wfo: pd.DataFrame, ret_1d: pd.Series,
             signal_col: str, regime_exit: bool = False) -> dict:
    pos = make_positions(wfo, signal_col, regime_exit)
    target_10d = wfo['target']
    r1 = ret_1d.reindex(wfo.index)

    pnl = pos * target_10d.fillna(0) / FORWARD_DAYS

    n_long  = (pos ==  1).sum()
    n_short = (pos == -1).sum()
    n_cash  = (pos ==  0).sum()
    n_tot   = len(pos)

    # Spearman IC между signal и realized 10d target
    vm = wfo[signal_col].notna() & target_10d.notna()
    ic, _ = spearmanr(wfo.loc[vm, signal_col], target_10d[vm])

    cum      = pnl.cumsum()
    bh       = r1.fillna(0).cumsum()
    sharpe   = pnl.mean() / (pnl.std() + eps) * np.sqrt(252)
    roll_max = cum.cummax()
    max_dd   = (cum - roll_max).min()
    win_rt   = (pnl[pos != 0] > 0).mean() if (pos != 0).any() else np.nan
    cum_ret  = cum.iloc[-1] * 100

    return dict(
        pos=pos, pnl=pnl, cum=cum, bh=bh,
        n_long=n_long, n_short=n_short, n_cash=n_cash, n_tot=n_tot,
        ic=ic, sharpe=sharpe, max_dd=max_dd, win_rate=win_rt,
        cum_ret=cum_ret, bh_ret=bh.iloc[-1]*100,
    )


# ─────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────
def print_comparison(results_val: dict, results_oos: dict):
    strategies = {
        'LightGBM':        ('p_lgb', False),
        'CatBoost':        ('p_cat', False),
        'Ensemble':        ('p_ens', False),
        'Ensemble+Regime': ('p_ens', True),
    }
    header = f"{'Strategy':<20} {'Sharpe_V':>8} {'MaxDD_V':>8} {'CumRet_V':>9} {'IC_V':>7} | " \
             f"{'Sharpe_O':>8} {'MaxDD_O':>8} {'CumRet_O':>9} {'IC_O':>7}"
    print('\n' + '='*85)
    print('COMPARISON TABLE  (V=VAL 2021-2024  O=OOS 2024-2026)')
    print('='*85)
    print(header)
    print('-'*85)
    best_sharpe_v = max(results_val[k]['sharpe'] for k in results_val)
    best_sharpe_o = max(results_oos[k]['sharpe'] for k in results_oos)
    for name, (col, re) in strategies.items():
        rv = results_val[name]
        ro = results_oos[name]
        sv  = f"{rv['sharpe']:+.3f}" + ('*' if rv['sharpe'] == best_sharpe_v else ' ')
        so  = f"{ro['sharpe']:+.3f}" + ('*' if ro['sharpe'] == best_sharpe_o else ' ')
        print(f"{name:<20} {sv:>9} {rv['max_dd']*100:>7.2f}% {rv['cum_ret']:>8.2f}% {rv['ic']:>7.4f} | "
              f"{so:>9} {ro['max_dd']*100:>7.2f}% {ro['cum_ret']:>8.2f}% {ro['ic']:>7.4f}")
    rv0 = list(results_val.values())[0]
    ro0 = list(results_oos.values())[0]
    print('-'*85)
    print(f"{'Buy & Hold':<20} {'':>9} {'':>8} {rv0['bh_ret']:>8.2f}% {'':>7} | "
          f"{'':>9} {'':>8} {ro0['bh_ret']:>8.2f}% {'':>7}")
    print('='*85)
    print('  * = best in column')


def print_trade_stats(name, rv, ro):
    print(f"\n  [{name}]")
    print(f"    VAL: Long={rv['n_long']}({100*rv['n_long']/rv['n_tot']:.0f}%)  "
          f"Short={rv['n_short']}({100*rv['n_short']/rv['n_tot']:.0f}%)  "
          f"Cash={rv['n_cash']}({100*rv['n_cash']/rv['n_tot']:.0f}%)  "
          f"WinRate={100*rv['win_rate']:.1f}%")
    print(f"    OOS: Long={ro['n_long']}({100*ro['n_long']/ro['n_tot']:.0f}%)  "
          f"Short={ro['n_short']}({100*ro['n_short']/ro['n_tot']:.0f}%)  "
          f"Cash={ro['n_cash']}({100*ro['n_cash']/ro['n_tot']:.0f}%)  "
          f"WinRate={100*ro['win_rate']:.1f}%")


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_all(wfo_val, wfo_oos, rv_map, ro_map, source_map, top_factors):
    colors = {
        'LightGBM':        '#2980B9',
        'CatBoost':        '#E74C3C',
        'Ensemble':        '#27AE60',
        'Ensemble+Regime': '#8E44AD',
    }

    # ── Chart 1: Cumulative returns ──────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 13),
                              gridspec_kw={'height_ratios': [3, 1.2, 1]})
    fig.suptitle('Monthly WFO — LightGBM vs CatBoost vs Ensemble + Regime Exit\n'
                 f'Top-{TOP_N_FACTORS} stable factors | 5yr window | 1m step | '
                 f'Q{int(Q_LONG*100)}/Q{int(Q_SHORT*100)} quantile | noise-p{NOISE_PCTILE}',
                 fontsize=11, y=0.99)

    ax = axes[0]
    bh_val_cum = rv_map['LightGBM']['bh']
    bh_oos_off = bh_val_cum.iloc[-1]
    bh_oos_cum = ro_map['LightGBM']['bh']

    # Buy & hold bridge
    (bh_val_cum * 100).plot(ax=ax, color='#7F8C8D', lw=1.5, ls=':', alpha=0.7, label='Buy & Hold')
    ((bh_oos_cum + bh_oos_off) * 100).plot(ax=ax, color='#7F8C8D', lw=1.5, ls=':', alpha=0.7)

    for name, c in colors.items():
        rv = rv_map[name]; ro = ro_map[name]
        # VAL
        (rv['cum'] * 100).plot(ax=ax, color=c, lw=2.0,
                                label=f"{name}  VAL={rv['cum_ret']:+.1f}%  OOS={ro['cum_ret']:+.1f}%")
        # OOS (offset from end of VAL)
        off = rv['cum'].iloc[-1]
        ((ro['cum'] + off) * 100).plot(ax=ax, color=c, lw=2.0, ls='--', alpha=0.75)

    ax.axvline(pd.Timestamp(OOS_START), color='red', lw=1.5, ls=':', label='OOS start')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('Cumulative Return (log, %)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(alpha=0.25)
    ax.set_title('Cumulative Returns: VAL (solid) + OOS (dashed)', fontsize=10)

    # ── Chart 2: Monthly active days ────────────────────────────────────
    ax2 = axes[1]
    ens_re_pos = pd.concat([rv_map['Ensemble+Regime']['pos'],
                             ro_map['Ensemble+Regime']['pos']])
    monthly_long  = ens_re_pos.clip(0).resample('ME').sum()
    monthly_short = (-ens_re_pos).clip(0).resample('ME').sum()

    ax2.bar(monthly_long.index,  monthly_long.values,
            color='#2980B9', alpha=0.7, width=20, label='Long days')
    ax2.bar(monthly_short.index, monthly_short.values,
            color='#E74C3C', alpha=0.7, width=20, bottom=monthly_long.values,
            label='Short days')
    ax2.axvline(pd.Timestamp(OOS_START), color='red', lw=1.2, ls=':')
    ax2.set_ylabel('Active days/month', fontsize=9)
    ax2.set_title('Ensemble+Regime Exit: Monthly Trading Activity', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.25)

    # ── Chart 3: Rolling IC per window ───────────────────────────────────
    ax3 = axes[2]
    both = pd.concat([wfo_val, wfo_oos])
    win_dates = both.groupby('window').apply(lambda g: g.index[0])
    ic_ens_s  = pd.Series(both.groupby('window')['ic_ens'].first().values, index=win_dates.values)
    ic_lgb_s  = pd.Series(both.groupby('window')['ic_lgb'].first().values, index=win_dates.values)
    ic_cat_s  = pd.Series(both.groupby('window')['ic_cat'].first().values, index=win_dates.values)

    ax3.plot(ic_lgb_s.index, ic_lgb_s.values, color='#2980B9', lw=1.0, alpha=0.7, label='LGB IC')
    ax3.plot(ic_cat_s.index, ic_cat_s.values, color='#E74C3C', lw=1.0, alpha=0.7, label='CatBoost IC')
    ax3.plot(ic_ens_s.index, ic_ens_s.values, color='#27AE60', lw=2.0, label='Ensemble IC')
    ax3.axhline(0, color='k', lw=0.8)
    ax3.axvline(pd.Timestamp(OOS_START), color='red', lw=1.2, ls=':')
    ax3.fill_between(ic_ens_s.index, ic_ens_s.values, 0,
                     where=ic_ens_s.values > 0, color='#27AE60', alpha=0.15)
    ax3.fill_between(ic_ens_s.index, ic_ens_s.values, 0,
                     where=ic_ens_s.values < 0, color='#E74C3C', alpha=0.15)
    ax3.set_ylabel('Spearman IC / window', fontsize=9)
    ax3.set_title('Rolling Window IC (monthly adaptation)', fontsize=10)
    ax3.legend(fontsize=8, loc='lower right', ncol=3)
    ax3.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('monthly_wfo_cumret.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved -> monthly_wfo_cumret.png')

    # ── Chart 2: Sharpe comparison bar ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (period, rmap) in enumerate([('VAL 2021-2024', rv_map),
                                            ('OOS 2024-2026', ro_map)]):
        names   = list(colors.keys())
        sharpes = [rmap[n]['sharpe'] for n in names]
        cols    = [colors[n] for n in names]
        bars = axes2[ax_i].bar(names, sharpes, color=cols, alpha=0.8, edgecolor='white', lw=1.5)
        axes2[ax_i].axhline(0, color='k', lw=0.8)
        axes2[ax_i].set_title(f'Sharpe Ratio — {period}', fontsize=11)
        axes2[ax_i].set_ylabel('Annualized Sharpe', fontsize=10)
        axes2[ax_i].grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, sharpes):
            axes2[ax_i].text(bar.get_x() + bar.get_width()/2,
                             v + 0.02 * np.sign(v) if v != 0 else 0.02,
                             f'{v:+.3f}', ha='center',
                             va='bottom' if v >= 0 else 'top', fontsize=9)
        axes2[ax_i].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig('monthly_wfo_sharpe.png', dpi=150)
    plt.close()
    print('  Saved -> monthly_wfo_sharpe.png')

    # ── Chart 3: Factor composition ───────────────────────────────────────
    fig3, ax4 = plt.subplots(figsize=(10, 8))
    n_show = min(40, len(top_factors))
    cols_f = ['#E74C3C' if source_map.get(n,'')=='intermarket' else '#2980B9'
              for n in top_factors[:n_show]]
    ax4.barh(range(n_show), [1]*n_show, color=cols_f, alpha=0.7)
    ax4.set_yticks(range(n_show))
    ax4.set_yticklabels(top_factors[:n_show], fontsize=7.5)
    ax4.set_title(f'Top-{TOP_N_FACTORS} Stable Factors Selected\n'
                  f'(red=intermarket, blue=single-asset)', fontsize=11)
    from matplotlib.patches import Patch
    ax4.legend(handles=[Patch(facecolor='#2980B9', label='Single-asset'),
                         Patch(facecolor='#E74C3C', label='Intermarket')], fontsize=9)
    plt.tight_layout()
    plt.savefig('monthly_wfo_factors.png', dpi=150)
    plt.close()
    print('  Saved -> monthly_wfo_factors.png')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    t0_total = time.time()

    print('=' * 70)
    print('MONTHLY WFO — LightGBM vs CatBoost vs Ensemble + Regime Exit')
    print('=' * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    print('\nStep 1 -- Load pool & features')
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool = json.load(f)
    pool_names = [r['name'] for r in pool]
    source_map = {r['name']: r.get('source','single_asset') for r in pool}

    full   = load_all_aligned()
    feat_df = build_feature_matrix(full, pool_names)
    close  = full['es_close']
    ret_1d = log_ret(close)
    target = np.log(close.shift(-FORWARD_DAYS) / close)
    print(f'  Feature matrix: {feat_df.shape}  |  '
          f'Data: {full.index[0].date()} -> {full.index[-1].date()}')

    # ── Factor selection ──────────────────────────────────────────────────
    print('\nStep 2 -- Stable factor selection')
    top_factors = select_stable_factors(pool, feat_df, target)
    n_im = sum(1 for n in top_factors if source_map.get(n,'')=='intermarket')
    print(f'  SA={TOP_N_FACTORS-n_im}  IM={n_im}  | top-5: {top_factors[:5]}')

    # ── Monthly WFO — VAL ─────────────────────────────────────────────────
    print(f'\nStep 3 -- Monthly WFO on VAL ({WFO_START} -> {WFO_END})')
    print(f'  window={TRAIN_WINDOW_YRS}yr  step={STEP_MONTHS}m  '
          f'Q={int(Q_LONG*100)}/{int(Q_SHORT*100)}  noise_p={NOISE_PCTILE}')
    wfo_val = rolling_wfo(feat_df, target, top_factors, WFO_START, WFO_END, 'VAL')

    # ── Monthly WFO — OOS ─────────────────────────────────────────────────
    print(f'\nStep 4 -- Monthly WFO on OOS ({OOS_START} -> {OOS_END})')
    wfo_oos = rolling_wfo(feat_df, target, top_factors, OOS_START, OOS_END, 'OOS')

    # ── Backtests ─────────────────────────────────────────────────────────
    print('\nStep 5 -- Running backtests (4 strategies)')
    strategies = {
        'LightGBM':        ('p_lgb', False),
        'CatBoost':        ('p_cat', False),
        'Ensemble':        ('p_ens', False),
        'Ensemble+Regime': ('p_ens', True),
    }
    rv_map, ro_map = {}, {}
    for name, (col, re) in strategies.items():
        rv_map[name] = backtest(wfo_val, ret_1d, col, re)
        ro_map[name] = backtest(wfo_oos, ret_1d, col, re)

    # ── Comparison table ──────────────────────────────────────────────────
    print_comparison(rv_map, ro_map)

    # ── Trade stats ───────────────────────────────────────────────────────
    print('\nTrade statistics:')
    for name in strategies:
        print_trade_stats(name, rv_map[name], ro_map[name])

    # ── Rolling IC summary ────────────────────────────────────────────────
    print(f'\nRolling IC:')
    print(f'  VAL  mean IC  LGB={wfo_val.groupby("window")["ic_lgb"].first().mean():+.4f}'
          f'  CAT={wfo_val.groupby("window")["ic_cat"].first().mean():+.4f}'
          f'  ENS={wfo_val.groupby("window")["ic_ens"].first().mean():+.4f}')
    print(f'  OOS  mean IC  LGB={wfo_oos.groupby("window")["ic_lgb"].first().mean():+.4f}'
          f'  CAT={wfo_oos.groupby("window")["ic_cat"].first().mean():+.4f}'
          f'  ENS={wfo_oos.groupby("window")["ic_ens"].first().mean():+.4f}')

    # ── Plots ─────────────────────────────────────────────────────────────
    print('\nStep 6 -- Generating charts...')
    plot_all(wfo_val, wfo_oos, rv_map, ro_map, source_map, top_factors)

    # ── Regime exit impact ────────────────────────────────────────────────
    re_val  = rv_map['Ensemble+Regime']
    ens_val = rv_map['Ensemble']
    re_oos  = ro_map['Ensemble+Regime']
    ens_oos = ro_map['Ensemble']
    print(f'\nRegime Exit impact:')
    print(f'  VAL: Ensemble {ens_val["cum_ret"]:+.2f}%  ->  +Regime {re_val["cum_ret"]:+.2f}%  '
          f'(Sharpe {ens_val["sharpe"]:+.3f} -> {re_val["sharpe"]:+.3f})'
          f'  Cash days: {ens_val["n_cash"]} -> {re_val["n_cash"]}')
    print(f'  OOS: Ensemble {ens_oos["cum_ret"]:+.2f}%  ->  +Regime {re_oos["cum_ret"]:+.2f}%  '
          f'(Sharpe {ens_oos["sharpe"]:+.3f} -> {re_oos["sharpe"]:+.3f})'
          f'  Cash days: {ens_oos["n_cash"]} -> {re_oos["n_cash"]}')

    elapsed = time.time() - t0_total
    print(f'\nTotal elapsed: {elapsed:.0f}s')
    print(f'Saved: monthly_wfo_cumret.png  monthly_wfo_sharpe.png  monthly_wfo_factors.png')
