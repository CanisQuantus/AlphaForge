#### В данной версии кода найдена ошибка заглядывания в будущее в meta модели. Код без ошибок будет выложен позднее.
# Chain of Alpha — Систематическая торговля фьючерсами на основе ML

**Автоматизированная система поиска и эксплуатации рыночных аномалий** на шести фьючерсных рынках. Проект охватывает полный цикл: от добычи альфа-факторов из сырых ценовых данных до управления многоактивным портфелем с динамическим плечом через мета-модель.

---

## Результаты вне выборки — OOS 2024–2026

Все результаты получены на данных **2024–2026**, которые ни разу не использовались в разработке. Модели обучались на 2011–2021, параметры валидировались на 2021–2024.

### Портфель ($100 000 капитал, 4x суммарное плечо)

| Модель          | Net P&L    | Sharpe | MaxDD   | Recovery | Profit Factor |
|:----------------|:----------:|:------:|:-------:|:--------:|:-------------:|
| Ridge Meta      | +$87 224   | +4.89  | −7.59%  | 11.49x   | 2.67          |
| CatBoost Meta   | +$85 893   | +5.05  | −7.26%  | 11.83x   | 2.77          |
| Equal-Weight    | +$91 953   | +4.87  | −7.23%  | 12.72x   | 2.64          |

### Отдельные активы (без плеча, Ensemble + Regime Exit)

| Актив               | Горизонт | Net P&L   | Sharpe | MaxDD   | Profit Factor |
|:--------------------|:--------:|:---------:|:------:|:-------:|:-------------:|
| S&P 500 (ES)        | 10d      | +$12 497  | +1.64  | −4.77%  | 1.50          |
| Nasdaq 100 (NQ)     | 15d      | +$15 626  | +2.02  | −7.42%  | 1.61          |
| Russell 2000 (RTY)  | 20d      | +$19 750  | +2.99  | −4.53%  | 2.19          |
| DAX (FDAX)          | 10d      | +$32 153  | +4.07  | −4.76%  | 2.68          |
| EuroStoxx 50 (FESX) | 10d      | +$20 631  | +2.44  | −4.84%  | 1.79          |
| Crude Oil (CL)      | 10d      | +$37 272  | +2.39  | −8.84%  | 1.80          |

> Комиссия: 0.006% RT (соответствует $1.50 за контракт на MES/MNQ/M2K при номинале ~$25k). Режимный фильтр включён для всех активов.

### Сравнение с бенчмарком (Buy & Hold, те же 6 активов, то же плечо 4x)

| Стратегия           |     Net P&L  | Sharpe   |   MaxDD  | Profit Factor |
|:--------------------|:------------:|:--------:|:--------:|:-------------:|
| **ML Equal-Weight** | **+$85 000** |**+4.46** |**−6.1%** |    **2.41**   |
| Buy & Hold          |  +$93 000    |  +0.73   | −64.1%   |      1.16     |

Простое удержание позиций на бычьем рынке 2024–2025 формально дало чуть больше денег — но ценой просадки −64% при 4x плече, что в реальности означает margin call. ML-стратегия уступает в абсолютном P&L примерно на 9%, но взамен даёт Sharpe в **6 раз выше** и просадку в **10 раз меньше** — то есть доходность, на которую можно реально опираться.

---

## Методология

### Общая схема

```
1-мин OHLCV (2011–2026)
       │
       ▼
Агрегация в дневные бары + VWAP
       │
       ▼
Генерация альфа-факторов
  ├─ Single-Asset (SA): математические операторы над ценой/объёмом
  └─ Intermarket (IM): сигналы относительно коррелированных партнёров
       │
       ▼
IC-фильтр (Spearman RankIC > порог) + dedup по корреляции
  → пул ~300 независимых факторов на актив
       │
       ▼
Отбор топ-50 стабильных факторов
  (критерий: sign_consistency × IC × доминирование знака по фолдам)
       │
       ▼
Monthly Walk-Forward Optimization
  ├─ LightGBM (L2, 300 деревьев)
  ├─ CatBoost (depth=5, l2=3)
  └─ Ансамбль: 50% LGB + 50% CAT
       │
       ▼
Позиции: Long top-30% | Short bottom-30% | Cash 40%
+ Режимный фильтр: cash если оба сигнала в зоне шума
       │
       ▼
Meta-Arbitrator ("Батя")
  ├─ Ridge Regression  (walk-forward)
  └─ CatBoost          (walk-forward)
  Предсказывает следующий Sharpe каждого актива → плечи 0.3x–1.2x
```

### Альфа-факторы

Система разделяет факторы на два класса:

- **Single-Asset (SA):** сигналы, построенные только на данных торгуемого инструмента. Включают структуру волатильности, паттерны объёма, ценовые каналы, моментум на различных горизонтах, отклонение от VWAP и другие. Окна от 5 до 120 дней.

- **Intermarket (IM):** сигналы, описывающие взаимосвязь инструмента с коррелированными рынками. Для каждого актива подобраны 7–9 партнёров среди американских и европейских индексов, облигаций, волатильности, валют и сырья.

Кандидатов в пул несколько тысяч. После IC-фильтра и жадного dedup по корреляции (|ρ| < 0.70) принимаются ~300 независимых факторов на актив.

### Горизонт предсказания

Оптимизируется отдельно для каждого актива в диапазоне **5–20 дней** на тренировочных данных с подтверждением на отдельном периоде. Это позволяет учесть различия в микроструктуре рынков (например, European index futures vs US equity index futures).

### Walk-Forward Optimization

- Скользящее обучающее окно: **5 лет**
- Шаг предсказания: **1 месяц**
- Квантильные пороги (long/short/cash) пересчитываются **внутри каждого WFO-окна** — система адаптируется к изменениям волатильности сигнала
- **Режимный фильтр:** если оба сигнала (LGB и CatBoost) находятся ниже noise-floor (p25 тренировочных предсказаний), позиция обнуляется. Защищает от торговли в условиях высокой неопределённости

### Meta-Arbitrator

Мета-модель работает поверх шести базовых стратегий. Каждый месяц предсказывает ожидаемый Sharpe для каждого актива на следующий месяц, используя ~60 признаков (IC, Sharpe за 1/3 месяца, волатильность, profit factor, коэффициент активности сигнала и т.д.). На основе предсказаний рассчитывается плечо:

- Ограничения: **0.3x–1.2x** на актив
- Суммарное плечо портфеля: **4.0x**
- Обучение: walk-forward, минимум 12 месяцев истории перед первым предсказанием

---

## Структура файлов

```
github/
├── global_pipeline.py        # Главный пайплайн: все 7 активов
├── meta_arbitrator.py        # Мета-модель и портфельный бэктест
├── monthly_wfo.py            # Ядро WFO (отдельный запуск для ES)
├── mes_final.py              # Финансовый враппер MES ($1.50 RT/контракт)
├── plot_portfolio_improved.py
├── plot_per_asset.py
├── risk_audit.py
└── results/
    ├── {TICKER}/
    │   ├── summary.json      # Метрики: Sharpe, MaxDD, Net Profit, IC
    │   └── horizon.txt       # Оптимальный горизонт предсказания
    ├── portfolio/
    │   └── oos_summary.csv
    └── meta/
        ├── per_asset_oos_breakdown.png
        └── meta_arbitrator_report.png
```

> Исходный код генерации сигналов, пулы факторов и обученные модели в публичный репозиторий не включены.

---

## Технический стек

`Python 3.11` · `LightGBM` · `CatBoost` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `matplotlib`

Данные: 1-минутные фьючерсные контракты от First Rate Data.

---
---

# Chain of Alpha — Systematic Futures Trading with ML

**Automated alpha discovery and portfolio management** across six futures markets. Covers the full research pipeline: from mining predictive signals in raw price data to dynamic portfolio allocation via a walk-forward meta-model.

---

## Out-of-Sample Results (2024–2026)

All results are from the **2024–2026 holdout period**, never used during development.

### Portfolio — $100k capital, 4x total leverage

| Model         | Net P&L  | Sharpe | Max DD  | Recovery | PF   |
|:--------------|:--------:|:------:|:-------:|:--------:|:----:|
| Ridge Meta    | +$87,224 | +4.89  | −7.59%  | 11.49x   | 2.67 |
| CatBoost Meta | +$85,893 | +5.05  | −7.26%  | 11.83x   | 2.77 |
| Equal-Weight  | +$91,953 | +4.87  | −7.23%  | 12.72x   | 2.64 |

### Individual assets (unlevered, Regime Exit ON)

| Asset               | Net P&L   | Sharpe | Max DD  | PF   |
|:--------------------|:---------:|:------:|:-------:|:----:|
| S&P 500 (ES)        | +$12,497  | +1.64  | −4.77%  | 1.50 |
| Nasdaq 100 (NQ)     | +$15,626  | +2.02  | −7.42%  | 1.61 |
| Russell 2000 (RTY)  | +$19,750  | +2.99  | −4.53%  | 2.19 |
| DAX (FDAX)          | +$32,153  | +4.07  | −4.76%  | 2.68 |
| EuroStoxx 50 (FESX) | +$20,631  | +2.44  | −4.84%  | 1.79 |
| Crude Oil (CL)      | +$37,272  | +2.39  | −8.84%  | 1.80 |

> Commission: 0.006% RT. Splits: Train 2011–2021 | Val 2021–2024 | OOS 2024–2026.

---

## Approach

The system generates hundreds of alpha factor candidates per asset — both single-asset (price/volume structure) and intermarket (cross-market relationships). Candidates pass an IC filter and correlation deduplication to form a pool of ~300 independent signals. The top 50 by a stability metric feed a monthly walk-forward ensemble of LightGBM and CatBoost. A regime filter zeros out positions when both models are below the noise floor. A meta-model (Ridge / CatBoost) then allocates dynamic leverage (0.3x–1.2x) across all six assets in a second walk-forward layer.

## Stack

`Python` · `LightGBM` · `CatBoost` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib`
