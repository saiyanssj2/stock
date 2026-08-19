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
