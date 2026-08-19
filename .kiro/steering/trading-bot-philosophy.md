---
inclusion: auto
---

# Trading Bot Development Philosophy

## Core Objective

Mục tiêu số 1: **Đánh bại thị trường** (beat the market).

Mọi quyết định thiết kế, kiến trúc, training, và tối ưu PHẢI hướng về mục tiêu này.

## Decision Framework

Khi có nhiều lựa chọn kỹ thuật, LUÔN chọn cách **tối ưu nhất trong giới hạn cho phép**:

### Giới hạn (Constraints)
- **Thời gian training:** Acceptable training time tùy thuộc vào improvement kỳ vọng
- **Phần cứng:** GPU VRAM, CPU, RAM hiện có
- **Data:** Số lượng mã, độ dài lịch sử, quality
- **Chi phí tiền = 0:** KHÔNG sử dụng dịch vụ trả phí (paid API, cloud GPU, premium data feeds). Chỉ dùng tài nguyên local và data miễn phí.

### Trade-off Rule
- Training time TĂNG → Model performance PHẢI TĂNG tương xứng
- Nếu approach A mất gấp đôi thời gian nhưng chỉ tốt hơn 1% → chọn approach nhanh hơn
- Nếu approach B mất gấp đôi thời gian nhưng tốt hơn 10%+ → chọn approach tốt hơn
- **Không chấp nhận** approach tốn resource mà KHÔNG cải thiện khả năng đánh bại thị trường

### Quality Hierarchy (ưu tiên từ cao đến thấp)
1. **Model accuracy / Sharpe ratio** — khả năng sinh lời
2. **Robustness** — ổn định qua nhiều market regimes
3. **Training efficiency** — tốc độ train
4. **Code elegance** — clean code

### Specific Principles
- Cross-symbol learning (học chéo nhiều mã) > Per-symbol isolated training (train riêng từng mã) khi dùng shared model
- Ensemble / multi-signal (kết hợp nhiều tín hiệu) > Single signal (1 tín hiệu duy nhất)
- Adaptive parameters (tham số tự thích ứng) > Fixed parameters (tham số cố định)
- Evidence-based decisions (quyết định dựa trên bằng chứng) > Gut feelings (cảm tính)
- Backtest validated (đã kiểm chứng qua backtest) > Theoretically optimal (tối ưu trên lý thuyết)

## Ngôn ngữ

- Khi dùng từ chuyên ngành tiếng Anh, LUÔN kèm giải thích tiếng Việt trong ngoặc.
- Ví dụ: "Sharpe ratio (tỷ lệ lợi nhuận/rủi ro)", "drawdown (mức sụt giảm tối đa)"
