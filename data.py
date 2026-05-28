import pandas as pd
from vnstock.api.quote import Quote

def get_ohlcv(symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
    q = Quote(symbol=symbol, source="VCI")
    df = q.history(start=start, end=end, interval=interval)
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df
