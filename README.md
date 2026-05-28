# 📈 Phân Tích Kỹ Thuật Chứng Khoán

## Cài đặt

```bash
cd D:\code\project\stock
python -m pip install -r requirements.txt
```

## Chạy app

```bash
cd D:\code\project\stock
run.bat
```

Trình duyệt tự mở tại `http://localhost:8501`

## Cấu trúc

| File | Mô tả |
|------|-------|
| `app.py` | Giao diện Streamlit chính |
| `analysis.py` | Tính toán chỉ báo kỹ thuật (EMA, RSI, MACD, BB...) |
| `data.py` | Lấy dữ liệu OHLCV từ vnstock |
| `news.py` | Tin tức VN/quốc tế, chỉ số thị trường |
| `scanner.py` | Quét toàn bộ ~1500 mã tìm tín hiệu MUA |
| `backtest.py` | Kiểm chứng bộ lọc với dữ liệu quá khứ |

## Tính năng

- **Thị trường thế giới**: S&P 500, Nasdaq, Dow Jones
- **Thị trường VN**: VNINDEX, VN30 với 5 chỉ số (EMA, RSI, MACD, ADX, OBV)
- **Tin tức**: Tin VN (CafeF, VnEconomy) + Quốc tế (Bloomberg, Reuters, CNBC) trong 24h
- **Bộ lọc mua**: Quét toàn thị trường theo RSI, MACD, EMA, Bollinger
- **Phân tích chi tiết**: Biểu đồ candlestick + chỉ báo kỹ thuật
- **Backtest**: Kiểm chứng bộ lọc tại ngày quá khứ, đánh giá T+5/T+10/T+20
