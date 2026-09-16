# Troubleshooting & Change Log

## 2026-08-07: Bổ sung confidence và cash vào kết quả backtest manual

### Vấn đề
- Kết quả backtest manual thiếu thông tin để hiểu tại sao model quyết định mua và mua bao nhiêu.

### Cách fix
1. Thêm 3 field vào `Trade` model (`models/backtest_models.py`):
   - `confidence`: Độ tin cậy của signal tại thời điểm mua (0-100%)
   - `cash_before`: Tiền mặt trước khi mua (VND)
   - `cash_after`: Tiền mặt sau khi mua (VND)

2. Sửa `_generate_signals_ml()` trả về cả signals và confidences
3. Sửa `_execute_trades()` để lưu confidence và cash vào Trade

### Files đã sửa
- `models/backtest_models.py` - Thêm 3 field vào Trade dataclass
- `engine/workers/backtest_worker.py`:
  - `_generate_signals()` → trả về `Tuple[List[str], List[float]]`
  - `_generate_signals_ml()` → tính confidence từ ML score
  - `_generate_signals_ema()` → tính confidence từ EMA diff
  - `_execute_trades()` → lưu confidence, cash_before, cash_after

### Cấu trúc Trade mới
| Field | Mô tả |
|-------|-------|
| symbol | Mã cổ phiếu |
| buy_date | Ngày mua |
| sell_date | Ngày bán |
| buy_price | Giá mua |
| sell_price | Giá bán |
| shares | Số lượng CP |
| pnl | Lãi/lỗ (VND) |
| pnl_pct | Lãi/lỗ (%) |
| holding_days | Số ngày giữ |
| **confidence** | Độ tin cậy (%) |
| **cash_before** | Tiền trước mua (VND) |
| **cash_after** | Tiền sau mua (VND) |

### Cách đọc kết quả
- `confidence = 80%` → Model khá chắc chắn về signal này
- `cash_before = 100M, cash_after = 5M` → Gần như all-in (95%)
- `cash_before = 100M, cash_after = 75M` → Mua thăm dò (25%)


---

## 2026-08-17: Fix Normalization inconsistency + loại trừ Index symbols

### Vấn đề 1: VNINDEX/VN30/HNX/UPCOM được trade trong backtest manual
- Đây là các **chỉ số thị trường**, không phải cổ phiếu
- Không thể mua/bán trực tiếp trên sàn
- Lọc VNINDEX chỉ hoạt động ở fallback, không áp dụng khi có `selected_symbols` trong preferences

### Cách fix
- Sửa `_get_symbols()` trong `page_backtest.py`:
  - Định nghĩa `INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNX", "UPCOM", "HNX30"}`
  - Loại trừ khỏi danh sách trade trong MỌI trường hợp (cả từ preferences lẫn fallback)

---

### Vấn đề 2: Backtest manual lỗ dù Training WR 80%

**Nguyên nhân gốc**: Normalization không nhất quán giữa training và inference

| Giai đoạn | Cách normalize cũ | Vấn đề |
|-----------|------------------|--------|
| Training (walk_forward.py) | Chỉ trên slice OOS (15%) | Model học với scale này |
| Backtest WF | Chỉ trên slice OOS (15%) | ✅ Khớp |
| Backtest Manual | Toàn bộ history 5+ năm | ❌ Scale khác hoàn toàn |
| Recommendations | Toàn bộ history | ❌ Scale khác |

**Ví dụ minh họa**:
```
FPT giá từ 120k → 80k trong 6 tháng

Normalize theo 6 tháng:  120k → 1.0,  80k → 0.0
Normalize theo 5 năm:    120k → 0.75, 80k → 0.42

→ Model nhận observation hoàn toàn khác → dự đoán sai
```

### Cách fix
Sửa backtest manual và recommendations để normalize **giống training** (chỉ trên period/window đang xét):

1. **`ui/pages/page_backtest.py`**:
   - Filter data theo period TRƯỚC
   - Normalize CHỈ trên period đã filter

2. **`engine/recommendation_engine.py`**:
   - Lấy lookback window TRƯỚC  
   - Normalize CHỈ trên window đó

### Bài học
- **Normalization phải NHẤT QUÁN** giữa training và inference
- Không quan trọng cách nào đúng về mặt học thuật, quan trọng là GIỐNG NHAU
- Model học từ data ở scale nào thì inference cũng phải ở scale đó

### Files đã sửa
- `ui/pages/page_backtest.py` - Normalize theo period backtest
- `engine/recommendation_engine.py` - Normalize theo lookback window


---

## 2026-08-17 (Update): Fix thiếu TCN Backbone trong Backtest Manual

### Vấn đề
Sau khi fix normalization, backtest manual vẫn cho WR ~39% (so với training 80%)

### Nguyên nhân gốc: Policy không có TCN Backbone

**Trong training (walk_forward.py)**:
```python
tcn_model = StockEvalNet(...)  # Train ở Stage 1
policy = PolicyNetwork(...)
policy.set_backbone(tcn_model)  # GẮN TCN VÀO POLICY
# → Policy encode features qua TCN → 64-dim → policy heads
```

**Trong backtest manual (cũ)**:
```python
policy = PolicyNetwork(...)
policy.load_state_dict(...)  # CHỈ LOAD WEIGHTS, KHÔNG SET BACKBONE!
# → Policy dùng FALLBACK FC layers → kết quả hoàn toàn khác!
```

### Cách fix
Sửa `page_backtest.py` để load cả TCN backbone:

```python
# 1. Load TCN model
tcn_model = StockEvalNet(model_config)
tcn_model.load_state_dict(...)

# 2. Tạo policy và GẮN BACKBONE
policy = PolicyNetwork(state_dim=state_dim)
policy.set_backbone(tcn_model, lookback=config.lookback)  # ← QUAN TRỌNG

# 3. Load policy weights
policy.load_state_dict(...)
```

### Files đã sửa
- `ui/pages/page_backtest.py` - Load TCN backbone trước khi load policy weights

### Bài học
- RL Policy có 2 mode: với backbone (TCN) và fallback (FC)
- Khi training dùng backbone → inference PHẢI dùng backbone
- Load state_dict KHÔNG tự động restore backbone reference


---

## 2026-08-19: Fix Training-Serving Skew trong Recommendation

### Triệu chứng
- Xem Recommendations → Model khuyến nghị **BÁN** một mã
- Chạy Backtest (1-2 cycles train)
- Xem lại Recommendations → Model khuyến nghị **GIỮ** cùng mã đó
- Inconsistency gây mất niềm tin vào model

### Nguyên nhân gốc
**Training-Serving Skew**: Training và Inference dùng normalization khác nhau:

1. **Training (trước fix)**: Per-window normalize mỗi slice 60 ngày
2. **Inference**: Per-window normalize chỉ trên lookback window hiện tại
3. **Problem**: Window hiện tại có thể có range khác (outliers, volatility) → model nhận input scale khác → prediction khác

Thêm vào đó, **model được update liên tục** khi training chạy, nên mỗi lần refresh recommendation có thể dùng model version khác.

### Cách fix: Feature Scaling Consistency

**Ý tưởng**: Fit **global scaler** trên toàn bộ training data, save params, inference dùng exact params đó.

#### Files đã tạo/sửa

| File | Thay đổi |
|------|----------|
| `engine/feature_scaler.py` | **Mới** - FeatureScaler class với fit/transform/save/load |
| `engine/wf_trainer/sl_trainer.py` | Dùng global scaler thay vì per-window normalize |
| `engine/wf_trainer/walk_forward.py` | Load existing scaler, pass vào train_supervised() |
| `engine/recommendation_engine.py` | Load saved scaler, dùng khi inference |
| `tests/unit/test_feature_scaler.py` | **Mới** - 34 unit tests |

#### Flow mới

```
Training (mỗi cycle):
1. build_sl_dataset() fit global scaler trên tất cả windows
2. train_supervised() save scaler vào engine/models/scaler_params.json
3. Model được train với features normalized theo global scaler

Inference (Recommendations):
1. _load_scaler() load params từ scaler_params.json (lazy, 1 lần)
2. _normalize_features() dùng saved scaler transform window
3. Model nhận input CÙNG scale như lúc train → prediction consistent
```

#### Fallback
Nếu `scaler_params.json` chưa có (chưa train cycle nào):
- Log warning
- Dùng per-window normalize như trước
- Sau khi train ít nhất 1 cycle → scaler có sẵn

#### Scaler file format
```json
{
  "min_vals": [0.1, 0.2, ...],  // 78 features
  "max_vals": [100.5, 200.3, ...],
  "num_features": 78,
  "feature_names": ["open", "high", ...],
  "fitted_at": "2026-08-19T10:30:00",
  "num_samples": 50000,
  "symbols_used": ["VNM", "FPT", "HPG", ...]
}
```

### Cách phòng ngừa
1. Luôn dùng **cùng normalization** giữa training và inference
2. **Save scaler params** cùng với model checkpoint
3. **Reload scaler** khi refresh recommendations
4. Test với `test_consistency_across_sessions` để verify

### Liên quan
- Best practice: [ML Trading Backtesting Guide](https://www.tradealgo.com/trading-guides/ai-trading/machine-learning-backtesting-guide)
- Concept: Training-Serving Skew, Feature Store
