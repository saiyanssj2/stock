# 📈 Stock Trading Platform - AI Decision Engine

Nền tảng phân tích và ra quyết định giao dịch chứng khoán Việt Nam,
sử dụng AI (TCN + Attention + RL) kết hợp phân tích kỹ thuật.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

## Chạy app

```bash
run.bat
```

Hoặc:
```bash
streamlit run app.py
```

Trình duyệt tự mở tại `http://localhost:8501`

## Chạy tests

```bash
pytest
```

## Cấu trúc project

```
├── app.py                  # Entry point - Streamlit + background pipeline
├── analysis.py             # Tính chỉ báo kỹ thuật (add_indicators)
├── run.bat                 # Script khởi chạy app
├── pipeline_debug.log      # Log pipeline runtime
├── IMPROVEMENTS.md         # Lịch sử cải thiện model
├── config/                 # Cấu hình (settings, market rules, preferences)
├── engine/                 # Core AI engine
│   ├── config.py           # ModelConfig, EngineConfig
│   ├── market_state.py     # INDICATOR_COLUMNS, FeatureVectorBuilder
│   ├── recommendation_engine.py  # Scan & recommend
│   ├── data_pipeline.py    # Update data từ vnstock
│   ├── wf_trainer/         # Walk-Forward training pipeline
│   │   ├── walk_forward.py # WF cycle orchestration
│   │   ├── sl_trainer.py   # Supervised Learning
│   │   ├── rl_agent.py     # RL policy network (PPO)
│   │   ├── rl_env.py       # Trading environment + reward
│   │   └── config.py       # WFConfig
│   ├── evaluation_model.py # StockEvalNet (TCN+Attention)
│   ├── models/             # Saved model checkpoints (.pt)
│   ├── strategies/         # Trading strategies
│   └── workers/            # Background workers
├── models/                 # Data models (dataclass)
├── orchestrator/           # Task orchestration
├── ui/                     # Streamlit UI
│   ├── pages/              # Multi-page: dashboard, recommend, settings...
│   └── components/         # Reusable UI components
├── tests/                  # Tests (unit, integration, property-based)
├── data/                   # Dữ liệu OHLCV (CSV per symbol)
└── .kiro/steering/         # Context files cho Kiro AI
```

## Pipeline tự động

Khi chạy app, pipeline background tự thực hiện:
1. **Update Data** — tải dữ liệu mới từ vnstock (theo slot M/N/P trong ngày)
2. **Walk-Forward Cycle** — SL train → RL fine-tune → Backtest out-of-sample
3. Lặp lại mỗi ~10 phút

## Model

- **Architecture:** TCN (Temporal Convolutional Network) + Multi-Head Attention
- **Input:** 78 features × 60 phiên lookback
- **Training:** Supervised Learning (cross-entropy) + Reinforcement Learning (REINFORCE with baseline)
- **Output:** Position score [-1, 1] cho mỗi mã

## Tính năng chính

- **AI Trading Signal**: TCN+Attention + RL cho quyết định BUY/SELL/HOLD
- **Walk-Forward Training**: Train liên tục, không overfit
- **Recommendations**: Quét watchlist, đưa khuyến nghị + confidence score
- **Portfolio Management**: Theo dõi vị thế, tiền mặt, P&L
- **Phân tích kỹ thuật**: 73 chỉ báo (trend, momentum, volatility, volume, Wyckoff)
- **Luật TTCK VN**: ±7% biên độ, T+2.5, lô 100 cổ phiếu
- **Real-time Dashboard**: Status, progress tracking, auto-refresh
