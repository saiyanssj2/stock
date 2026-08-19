import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # === TREND ===
    df["EMA_9"]  = ta.trend.ema_indicator(close, window=9)
    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    df["EMA_50"] = ta.trend.ema_indicator(close, window=50)
    df["EMA_200"]= ta.trend.ema_indicator(close, window=200)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)

    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"]     = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()

    aroon = ta.trend.AroonIndicator(high, low, window=25)
    df["Aroon_up"]   = aroon.aroon_up()
    df["Aroon_down"] = aroon.aroon_down()

    df["CCI_20"] = ta.trend.cci(high, low, close, window=20)

    psar = ta.trend.PSARIndicator(high, low, close)
    df["PSAR"]      = psar.psar()
    df["PSAR_up"]   = psar.psar_up()
    df["PSAR_down"] = psar.psar_down()

    ich = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    df["Ichimoku_conv"]  = ich.ichimoku_conversion_line()
    df["Ichimoku_base"]  = ich.ichimoku_base_line()
    df["Ichimoku_a"]     = ich.ichimoku_a()
    df["Ichimoku_b"]     = ich.ichimoku_b()

    # === MOMENTUM ===
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    df["RSI_7"]  = ta.momentum.rsi(close, window=7)

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["STOCH_k"] = stoch.stoch()
    df["STOCH_d"] = stoch.stoch_signal()

    stochrsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["StochRSI_k"] = stochrsi.stochrsi_k()
    df["StochRSI_d"] = stochrsi.stochrsi_d()

    df["Williams_R"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["ROC_10"]     = ta.momentum.roc(close, window=10)
    df["TSI"]        = ta.momentum.tsi(close, window_slow=25, window_fast=13)
    df["UO"]         = ta.momentum.ultimate_oscillator(high, low, close)
    df["AO"]         = ta.momentum.awesome_oscillator(high, low)

    # === VOLATILITY ===
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_pband"]  = bb.bollinger_pband()
    df["BB_wband"]  = bb.bollinger_wband()

    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["ATR_7"]  = ta.volatility.average_true_range(high, low, close, window=7)

    kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
    df["KC_upper"]  = kc.keltner_channel_hband()
    df["KC_middle"] = kc.keltner_channel_mband()
    df["KC_lower"]  = kc.keltner_channel_lband()

    dc = ta.volatility.DonchianChannel(high, low, close, window=20)
    df["DC_upper"]  = dc.donchian_channel_hband()
    df["DC_middle"] = dc.donchian_channel_mband()
    df["DC_lower"]  = dc.donchian_channel_lband()

    df["Ulcer_14"] = ta.volatility.ulcer_index(close, window=14)

    # === VOLUME ===
    df["OBV"]   = ta.volume.on_balance_volume(close, volume)
    df["MFI"]   = ta.volume.money_flow_index(high, low, close, volume, window=14)
    df["CMF"]   = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    df["VWAP"]  = ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)
    df["ADI"]   = ta.volume.acc_dist_index(high, low, close, volume)
    df["FI"]    = ta.volume.force_index(close, volume, window=13)
    df["VPT"]   = ta.volume.volume_price_trend(close, volume)
    df["EOM"]   = ta.volume.ease_of_movement(high, low, volume, window=14)
    df["NVI"]   = ta.volume.negative_volume_index(close, volume)

    # === WYCKOFF / VOLUME-PRICE ANALYSIS (5 features) ===
    import numpy as np

    # 1. Effort vs Result: volume_change / price_change
    # Volume tăng mà giá ít đi → tín hiệu đảo chiều (absorption)
    price_range = (high - low).replace(0, np.nan)
    vol_sma20 = volume.rolling(20).mean()
    df["WK_effort_result"] = (volume / vol_sma20) / (price_range / price_range.rolling(20).mean())

    # 2. Volume Climax: volume / SMA_volume_20
    # Ngày volume đột biến → có thể là climax (đỉnh/đáy)
    df["WK_vol_climax"] = volume / vol_sma20

    # 3. Spread Position: vị trí close trong range ngày (0=đáy, 1=đỉnh)
    # Close gần high + volume cao → demand. Close gần low + volume cao → supply
    df["WK_spread_pos"] = (close - low) / price_range

    # 4. Spring/Upthrust: giá xuyên support/resistance rồi quay lại
    # (low - support_20d) / ATR — âm = spring (xuyên xuống rồi quay lên)
    support_20 = low.rolling(20).min()
    atr_14 = df["ATR_14"] if "ATR_14" in df.columns else price_range.rolling(14).mean()
    df["WK_spring"] = (low - support_20) / atr_14.replace(0, np.nan)

    # 5. Volume Trend: OBV slope (OBV tăng = accumulation, OBV giảm = distribution)
    obv = df["OBV"]
    df["WK_obv_slope"] = (obv - obv.rolling(10).mean()) / obv.rolling(10).std().replace(0, np.nan)

    # === VALUE INVESTING PROXIES (6 features) ===
    # Mô phỏng hành vi "giá trị" từ dữ liệu giá — cổ phiếu rẻ/đắt relative

    # 1. Mean Reversion: độ lệch giá so với SMA200 — xa = oversold/overbought
    sma_200 = df["EMA_200"] if "EMA_200" in df.columns else close.rolling(200).mean()
    df["VAL_mean_reversion"] = (close - sma_200) / sma_200.replace(0, np.nan)

    # 2. 52-Week Position: vị trí giá trong 250 phiên (0=đáy, 1=đỉnh)
    high_250 = close.rolling(250, min_periods=60).max()
    low_250 = close.rolling(250, min_periods=60).min()
    range_250 = (high_250 - low_250).replace(0, np.nan)
    df["VAL_52w_position"] = (close - low_250) / range_250

    # 3. Drawdown: % giảm từ đỉnh 250 phiên — drawdown lớn = có thể oversold
    rolling_max_250 = close.rolling(250, min_periods=60).max()
    df["VAL_drawdown"] = (close - rolling_max_250) / rolling_max_250.replace(0, np.nan)

    # 4. Recovery Ratio: tốc độ phục hồi trong 60 phiên
    high_60 = close.rolling(60, min_periods=20).max()
    low_60 = close.rolling(60, min_periods=20).min()
    range_60 = (high_60 - low_60).replace(0, np.nan)
    df["VAL_recovery_ratio"] = (close - low_60) / range_60

    # 5. Volatility Contraction: ATR ngắn / ATR dài — giảm = sắp breakout
    atr_7_val = df["ATR_7"] if "ATR_7" in df.columns else price_range.rolling(7).mean()
    atr_14_val = df["ATR_14"] if "ATR_14" in df.columns else price_range.rolling(14).mean()
    df["VAL_vol_contraction"] = atr_7_val / atr_14_val.replace(0, np.nan)

    # 6. Smart Accumulation: volume thấp + giá tăng = smart money accumulation
    vol_sma5 = volume.rolling(5).mean()
    vol_sma60 = volume.rolling(60, min_periods=20).mean()
    vol_ratio = vol_sma5 / vol_sma60.replace(0, np.nan)
    price_dir = close.pct_change(5).clip(-1, 1)  # hướng giá 5 phiên
    # Accumulation = volume thấp (ratio<1) kèm giá tăng nhẹ
    df["VAL_smart_accumulation"] = price_dir / vol_ratio.replace(0, np.nan)

    # === MARKET CONTEXT (6 features — VNINDEX + VN30) ===
    # Thông tin thị trường chung giúp model biết bull/bear market
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else "."
    # Thử đọc từ thư mục data/
    for _data_path in ["data", ".", os.path.join(data_dir, "data")]:
        _vnindex_path = os.path.join(_data_path, "VNINDEX.csv")
        _vn30_path = os.path.join(_data_path, "VN30.csv")
        if os.path.exists(_vnindex_path):
            break

    # VNINDEX context (3 features)
    try:
        if os.path.exists(_vnindex_path):
            _vni = pd.read_csv(_vnindex_path, usecols=["time", "close", "volume"])
            _vni["time"] = pd.to_datetime(_vni["time"])
            _vni = _vni.set_index("time")
            # Align theo time index của df
            df_time = pd.to_datetime(df["time"])
            _vni_close = df_time.map(_vni["close"]).astype(float)
            _vni_vol = df_time.map(_vni["volume"]).astype(float)
            # Feature 1: VNINDEX return 5 ngày (thị trường chung tăng/giảm)
            df["MKT_vni_ret5"] = _vni_close.pct_change(5).values
            # Feature 2: VNINDEX volume ratio (thanh khoản thị trường)
            _vni_vol_sma = _vni_vol.rolling(20).mean()
            df["MKT_vni_vol_ratio"] = (_vni_vol / _vni_vol_sma).values
            # Feature 3: Stock vs VNINDEX (outperform/underperform)
            _stock_ret5 = close.pct_change(5)
            _mkt_ret5 = _vni_close.pct_change(5)
            df["MKT_vs_vni"] = (_stock_ret5 - _mkt_ret5.values).values
        else:
            df["MKT_vni_ret5"] = 0.0
            df["MKT_vni_vol_ratio"] = 1.0
            df["MKT_vs_vni"] = 0.0
    except Exception:
        df["MKT_vni_ret5"] = 0.0
        df["MKT_vni_vol_ratio"] = 1.0
        df["MKT_vs_vni"] = 0.0

    # VN30 context (3 features)
    try:
        if os.path.exists(_vn30_path):
            _vn30 = pd.read_csv(_vn30_path, usecols=["time", "close", "volume"])
            _vn30["time"] = pd.to_datetime(_vn30["time"])
            _vn30 = _vn30.set_index("time")
            _vn30_close = df_time.map(_vn30["close"]).astype(float)
            _vn30_vol = df_time.map(_vn30["volume"]).astype(float)
            # Feature 4: VN30 return 5 ngày (blue-chip momentum)
            df["MKT_vn30_ret5"] = _vn30_close.pct_change(5).values
            # Feature 5: VN30 volume ratio
            _vn30_vol_sma = _vn30_vol.rolling(20).mean()
            df["MKT_vn30_vol_ratio"] = (_vn30_vol / _vn30_vol_sma).values
            # Feature 6: Stock vs VN30 (so với blue-chip)
            _mkt30_ret5 = _vn30_close.pct_change(5)
            df["MKT_vs_vn30"] = (_stock_ret5 - _mkt30_ret5.values).values
        else:
            df["MKT_vn30_ret5"] = 0.0
            df["MKT_vn30_vol_ratio"] = 1.0
            df["MKT_vs_vn30"] = 0.0
    except Exception:
        df["MKT_vn30_ret5"] = 0.0
        df["MKT_vn30_vol_ratio"] = 1.0
        df["MKT_vs_vn30"] = 0.0

    return df
