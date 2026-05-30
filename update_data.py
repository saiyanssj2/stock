from vnstock.api.quote import Quote
from analysis import add_indicators
import pandas as pd
import sys

def update_csv(symbol, path, start='2020-01-01', end='2099-12-31'):
    from datetime import date
    end = min(end, str(date.today()))

    q = Quote(symbol=symbol, source='VCI')
    df_new = q.history(start=start, end=end, interval='1D')
    if df_new is None or len(df_new) == 0:
        print(f'{symbol}: khong co du lieu moi')
        return

    df_new.columns = [c.lower() for c in df_new.columns]
    df_new['time'] = pd.to_datetime(df_new['time'])

    import os
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_old['time'] = pd.to_datetime(df_old['time'])
        base = ['time','open','high','low','close','volume']
        df = pd.concat([df_old[base], df_new[base]], ignore_index=True)
    else:
        df = df_new[['time','open','high','low','close','volume']].copy()

    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    df = add_indicators(df)
    df.round(4).to_csv(path, index=False)
    print(f'{symbol}: {len(df)} phien, den {df.time.iloc[-1].date()}, gia {df.close.iloc[-1]}, cols={len(df.columns)}')

# Mac dinh cap nhat VIX va VNINDEX
symbols = {
    'VIX':     r'D:\code\project\stock\VIX.csv',
    'VNINDEX': r'D:\code\project\stock\VNINDEX.csv',
}

# Cho phep truyen tham so: python update_data.py VNM
if len(sys.argv) > 1:
    sym = sys.argv[1].upper()
    path = rf'D:\code\project\stock\{sym}.csv'
    symbols = {sym: path}

for sym, path in symbols.items():
    update_csv(sym, path)

print('Done')
