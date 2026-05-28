import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, date, timedelta
from vnstock.api.quote import Quote
import ta

FEEDS_VN = {
    "CafeF": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "VnEconomy": "https://vneconomy.vn/chung-khoan.rss",
    "Vietstock": "https://vietstock.vn/830/chung-khoan.htm",
}

FEEDS_INTL = {
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "CNBC": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
}

INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
}

def _fetch_today(feeds: dict, limit: int) -> list[dict]:
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    items = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                if not (hasattr(e, "published_parsed") and e.published_parsed):
                    continue
                pub_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
                local_dt = pub_dt.astimezone()
                items.append({
                    "source": source,
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "published": local_dt.strftime("%d/%m %H:%M"),
                })
        except Exception:
            continue
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]

def get_news_vn(limit: int = 15) -> list[dict]:
    return _fetch_today(FEEDS_VN, limit)

def get_news_intl(limit: int = 10) -> list[dict]:
    return _fetch_today(FEEDS_INTL, limit)

VN_INDICES = ["VNINDEX", "VN30", "HNX", "UPCOM"]

def _score_market(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema20 = ta.trend.ema_indicator(close, window=20).iloc[-1]
    ema50 = ta.trend.ema_indicator(close, window=50).iloc[-1]
    rsi   = ta.momentum.rsi(close, window=14).iloc[-1]

    macd_obj  = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    macd_val  = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    adx_val = adx.adx().iloc[-1]
    adx_pos = adx.adx_pos().iloc[-1]   # +DI
    adx_neg = adx.adx_neg().iloc[-1]   # -DI

    obv = ta.volume.on_balance_volume(close, df["volume"])
    obv_ema = ta.trend.ema_indicator(obv, window=20)
    obv_trend = obv.iloc[-1] > obv_ema.iloc[-1]  # True = dong tien vao

    score = 0
    details = []

    # EMA cross
    if ema20 > ema50:
        score += 1
        details.append(f"EMA20 > EMA50")
    else:
        score -= 1
        details.append(f"EMA20 < EMA50")

    # RSI
    if rsi > 55:
        score += 1
        details.append(f"RSI={rsi:.0f} (mạnh)")
    elif rsi < 45:
        score -= 1
        details.append(f"RSI={rsi:.0f} (yếu)")
    else:
        details.append(f"RSI={rsi:.0f} (trung tính)")

    # MACD
    if macd_val > macd_sig:
        score += 1
        details.append("MACD > Signal")
    else:
        score -= 1
        details.append("MACD < Signal")

    # ADX - xu huong co manh khong
    if adx_val > 25:
        if adx_pos > adx_neg:
            score += 1
            details.append(f"ADX={adx_val:.0f} (trend tăng mạnh)")
        else:
            score -= 1
            details.append(f"ADX={adx_val:.0f} (trend giảm mạnh)")
    else:
        details.append(f"ADX={adx_val:.0f} (sideway)")

    # OBV - dong tien
    if obv_trend:
        score += 1
        details.append("OBV > EMA (tiền vào)")
    else:
        score -= 1
        details.append("OBV < EMA (tiền ra)")

    if score >= 3:
        trend = "🟢 Tăng"
    elif score <= -2:
        trend = "🔴 Giảm"
    else:
        trend = "🟡 Trung lập"

    return {
        "trend": trend,
        "score": score,
        "details": details,
        "ema20": ema20, "ema50": ema50,
        "rsi": rsi,
        "macd": macd_val, "macd_signal": macd_sig,
        "adx": adx_val,
    }

def get_vn_indices() -> list[dict]:
    end = str(date.today())
    start = str(date.today() - timedelta(days=120))
    result = []
    for symbol in VN_INDICES:
        try:
            q = Quote(symbol=symbol, source="VCI")
            df = q.history(start=start, end=end, interval="1D")
            if df is None or len(df) < 30:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values("time").reset_index(drop=True)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            change = last["close"] - prev["close"]
            pct = change / prev["close"] * 100

            m = _score_market(df)
            result.append({
                "name": symbol,
                "close": last["close"],
                "change": change,
                "pct": pct,
                **m,
            })
        except Exception:
            continue
    return result

def get_world_indices() -> list[dict]:
    result = []
    for name, symbol in INDICES.items():
        try:
            df = yf.Ticker(symbol).history(period="2d")
            if len(df) < 2:
                continue
            prev_close = df["Close"].iloc[-2]
            last_close = df["Close"].iloc[-1]
            change = last_close - prev_close
            pct = change / prev_close * 100
            result.append({"name": name, "price": last_close, "change": change, "pct": pct})
        except Exception:
            continue
    return result
