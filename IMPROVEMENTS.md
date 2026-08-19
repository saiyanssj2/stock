# Model Improvements Log

Ghi chép các cải thiện đã thử, kết quả, và kế hoạch tương lai.

---

## Baseline

- **Features:** 66 (5 OHLCV + 56 Technical + 5 Wyckoff)
- **Architecture:** TCN + Attention
- **RL:** 300 episodes, PPO-style
- **WR trung bình:** ~51.0% (cycles #83-#102)
- **RL avg_return:** ~19.2%
- **WR cao nhất:** 52.3%

---

## Đã thử

### 1. Market Context Features (VNINDEX + VN30) — ❌ KHÔNG HIỆU QUẢ
- **Thời gian:** Cycles #129-#148 (72 features)
- **Nội dung:** Thêm 6 features: VNI return 5d, VNI volume ratio, Stock vs VNI, VN30 return 5d, VN30 volume ratio, Stock vs VN30
- **Kết quả:** WR 50.7% (baseline 51.0%), RL avg 18.7% (baseline 19.2%)
- **Kết luận:** Model đã học gián tiếp thị trường chung qua price patterns. Thêm trực tiếp không tạo signal mới.
- **Trạng thái:** Giữ trong code (không rollback), không gây hại

### 2. Value Investing Proxies — ❌ KHÔNG HIỆU QUẢ
- **Thời gian:** Cycles #165-#191 (78 features)
- **Nội dung:** Thêm 6 features: mean_reversion, 52w_position, drawdown, recovery_ratio, vol_contraction, smart_accumulation
- **Kết quả:** WR 50.4%, RL avg 17.0% (thấp hơn baseline)
- **Kết luận:** Chỉ là biến đổi toán học của price data đã có (SMA, ATR, rolling). Không thêm thông tin mới.
- **Trạng thái:** Đang active, cần rollback hoặc giữ

### 3. Wyckoff Volume-Price Analysis — ✅ ĐÃ ÁP DỤNG (từ trước)
- **Nội dung:** 5 features: effort_result, vol_climax, spread_pos, spring, obv_slope
- **Kết quả:** Đã tích hợp vào baseline, đóng góp vào val_loss thấp hơn
- **Trạng thái:** Active, giữ nguyên

### 4. Cải thiện khác (từ conversation trước) — ✅ ĐÃ ÁP DỤNG
- Chi tiết không rõ từ context transfer, nhưng đã được áp dụng trước baseline #83
- **Trạng thái:** Active

---

## Đang thực hiện

### 6. Confidence Threshold trong Backtest — ❌ THẤT BẠI
- **Vấn đề gốc:** Muốn lọc bỏ trades kém chất lượng để tăng WR
- **Thử 1:** Greedy argmax + threshold 40% → 0 trades (HOLD luôn là argmax)
- **Thử 2:** Trade khi prob > 20% VÀ > HOLD prob → vẫn gần 0 trades
- **Nguyên nhân:** Policy phân phối probability: HOLD ~30-40%, mỗi trade action ~10-15%. Không action nào vượt HOLD → filter block hết.
- **Kết luận:** Confidence threshold KHÔNG tương thích với policy sampling hiện tại. Cần thay đổi architecture (separate buy/sell head) hoặc train policy khác mới áp dụng được.
- **Trạng thái:** ĐÃ ROLLBACK về sampling cũ
- **Backup:** `stock_eval_net_backup_pre_confidence.pt` + `rl_policy_backup_pre_confidence.pt`

### 5. Cải thiện RL Reward Function v2 — ❌ THẤT BẠI
- **Vấn đề gốc:** RL avg giảm từ 21% → 17%, muốn agent trade chọn lọc hơn
- **Thay đổi:** Over-trading penalty 0.0001→0.002, drawdown 0.1→0.15, hold bonus, risk penalty
- **Kết quả (cycles #230-#237):** RL avg giảm tiếp xuống ~10%, WR giảm 50.1%, backtest giảm ~1950%
- **Nguyên nhân:** Penalty quá nặng → agent bị "sợ" trade → return giảm mà trades không giảm
- **Trạng thái:** ĐÃ ROLLBACK về reward v1

---

## Kế hoạch tương lai

### 6. Tăng RL Episodes + Exploration
- Tăng 300 → 500-1000 episodes
- Thêm entropy bonus
- Decay learning rate

### 7. Trading Logic / Confidence Threshold
- Chỉ trade khi confidence > threshold
- Stop-loss / take-profit dùng ATR
- Lọc bớt trade chất lượng thấp

### 8. Architecture Changes
- TCN deeper / wider
- Bigger attention heads
- Ensemble nhiều model

### 9. Fundamental Data (nếu có nguồn)
- P/E, P/B, ROE, EPS từ vnstock API
- Map quarterly → daily
- Phức tạp nhưng tiềm năng lớn

### 10. Label Engineering
- Thay đổi cách tính target cho SL
- Dynamic horizon thay vì fixed 5 ngày
- Peak detection cho label tốt hơn

---

## Ghi chú kỹ thuật

- **Backup models:** `engine/models/stock_eval_net_backup_66features.pt`, `engine/models/stock_eval_net_backup_72features.pt`
- **Config locations:** `engine/config.py`, `engine/wf_trainer/config.py`, `engine/market_state.py`
- **Feature pipeline:** `analysis.py` → `add_indicators()`
- **RL code:** `engine/wf_trainer/rl_agent.py`, `engine/wf_trainer/rl_env.py`
