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

    return df
