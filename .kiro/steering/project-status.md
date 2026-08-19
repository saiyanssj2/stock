---
inclusion: auto
---

# Project Status & Context

File này cung cấp context cho Kiro mỗi session mới. Cập nhật sau mỗi thay đổi quan trọng.

**Cập nhật lần cuối:** 2026-07-13

---

## Architecture

- **App:** Streamlit multi-page (`app.py`), pipeline chạy background thread
- **Model:** TCN + Attention (Supervised Learning) → RL fine-tune (PPO/REINFORCE)
- **Training:** Walk-Forward cycles, auto mỗi ~10 phút
- **Data:** OHLCV từ vnstock API, lưu CSV trong `data/`

## Model hiện tại

- **Features:** 78 (5 OHLCV + 73 indicators)
  - 56 Technical (EMA, RSI, MACD, BB, ATR, OBV...)
  - 5 Wyckoff (effort_result, vol_climax, spread_pos, spring, obv_slope)
  - 6 Value Proxies (mean_reversion, 52w_position, drawdown, recovery_ratio, vol_contraction, smart_accumulation)
  - 6 Market Context (VNINDEX + VN30 returns, volume ratio, relative performance)
- **Lookback:** 60 phiên
- **RL:** 300 episodes, 6 actions (HOLD, BUY 25/50/100%, SELL 50/100%)
- **Reward:** daily_return - drawdown_penalty(0.1) - action_penalty(0.0001)
- **Win Rate hiện tại:** ~50.5% (plateau)

## Key Files

| File | Vai trò |
|------|---------|
| `app.py` | Entry point, pipeline background worker |
| `analysis.py` | Tính indicators (add_indicators) |
| `engine/config.py` | ModelConfig (num_features, architecture) |
| `engine/market_state.py` | INDICATOR_COLUMNS, FeatureVectorBuilder |
| `engine/wf_trainer/` | Walk-Forward: sl_trainer, rl_agent, rl_env, walk_forward |
| `engine/wf_trainer/config.py` | WFConfig (num_features, rl_episodes, etc.) |
| `engine/wf_trainer/rl_env.py` | TradingEnv + Reward function |
| `engine/recommendation_engine.py` | Scan & recommend mã |
| `config/preferences.py` | Load user preferences (symbols, capital) |
| `IMPROVEMENTS.md` | Log tất cả cải thiện đã thử & kết quả |
| `pipeline_debug.log` | Log pipeline runtime |

## Improvements History (tóm tắt)

Chi tiết đầy đủ: xem #[[file:IMPROVEMENTS.md]]

| # | Cải thiện | Kết quả |
|---|-----------|---------|
| 1 | Market Context (VNINDEX+VN30) | ❌ Không hiệu quả |
| 2 | Value Investing Proxies | ❌ Không hiệu quả |
| 3 | Wyckoff features | ✅ Đã áp dụng |
| 4 | Reward Function v2 (over-trading penalty, hold bonus) | ❌ Thất bại, đã rollback |

## Hướng tiếp theo (TODO)

1. Đánh giá Reward v2 (cần ~10 cycles)
2. Tăng RL episodes (300 → 500-1000) nếu reward v2 tốt
3. Confidence threshold (chỉ trade khi model tin > X%)
4. Label engineering (thay cách tính target SL)
5. Fundamental data (P/E, ROE — cần nguồn data mới)

## Lưu ý khi sửa code

- Sau khi thay đổi features: cập nhật `INDICATOR_COLUMNS` (market_state.py), `num_features` (config.py + wf_trainer/config.py)
- **BACKUP/ROLLBACK BẮT BUỘC:** Luôn backup VÀ rollback CẢ 2 file cùng lúc:
  - `engine/models/stock_eval_net.pt` (SL model)
  - `engine/models/rl_policy.pt` (RL policy)
  - Lý do: RL policy phụ thuộc SL backbone. Rollback riêng 1 file sẽ gây mismatch, cần nhiều cycles để re-adapt.
  - Naming convention: `*_backup_<mô_tả>.pt` (ví dụ: `stock_eval_net_backup_72features.pt`)
- Pipeline log: `pipeline_debug.log` (root)
- Mỗi cycle SL mất ~30-60 phút, RL ~8 phút, Backtest ~2 phút
- App chạy bằng `run.bat`, pipeline tự trigger mỗi 10 phút
