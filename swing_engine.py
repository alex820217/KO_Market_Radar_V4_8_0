import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

def _download(ticker, period, interval):
    df=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    return df.dropna(how='all')

def _ind(df):
    x=df.copy()
    for n in (20,50,200): x[f'EMA{n}']=EMAIndicator(x.Close,n).ema_indicator()
    x['RSI']=RSIIndicator(x.Close,14).rsi();x['MACD']=MACD(x.Close).macd_diff()
    x['ADX']=ADXIndicator(x.High,x.Low,x.Close,14).adx();x['ATR']=AverageTrueRange(x.High,x.Low,x.Close,14).average_true_range()
    x['RVOL']=x.Volume/x.Volume.rolling(20).mean() if 'Volume' in x else 1.0
    x['LOW10']=x.Low.rolling(10).min();x['HIGH10']=x.High.rolling(10).max()
    return x.dropna()

def scan_swing_asset(meta):
    h1=_download(meta['ticker'],'730d','1h')
    d1=_download(meta['ticker'],'3y','1d')
    if len(h1)<300 or len(d1)<220: raise ValueError('zu wenig Daten')
    h1=h1.iloc[:-1];d1=d1.iloc[:-1]
    h4=h1.resample('4h',origin='start_day').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4=_ind(h4);d1=_ind(d1)
    if h4.empty or d1.empty: raise ValueError('Indikatoren unvollständig')
    h=h4.iloc[-1];d=d1.iloc[-1]
    def score(long=True):
        tests=[(d.Close>d.EMA200 if long else d.Close<d.EMA200,15),(d.EMA20>d.EMA50 if long else d.EMA20<d.EMA50,10),(d.MACD>0 if long else d.MACD<0,10),(h.Close>h.EMA50 if long else h.Close<h.EMA50,15),(h.EMA20>h.EMA50 if long else h.EMA20<h.EMA50,10),(52<=h.RSI<=68 if long else 32<=h.RSI<=48,10),(h.MACD>0 if long else h.MACD<0,10),(h.ADX>=20,10),(abs(h.Close-h.EMA20)<=h.ATR,5),(h.RVOL>=1,5)]
        return sum(p for ok,p in tests if ok)
    ls,ss=score(True),score(False);direction='LONG' if ls>=ss else 'SHORT';score_=max(ls,ss)
    daily='LONG' if d.Close>d.EMA200 and d.EMA20>d.EMA50 else 'SHORT' if d.Close<d.EMA200 and d.EMA20<d.EMA50 else 'NEUTRAL'
    bias='LONG' if h.Close>h.EMA50 and h.EMA20>h.EMA50 else 'SHORT' if h.Close<h.EMA50 and h.EMA20<h.EMA50 else 'NEUTRAL'
    entry=float(h.Close);dist=abs(h.Close-h.EMA20)/h.ATR if h.ATR else 99
    stop=float(min(h.LOW10,entry-1.5*h.ATR) if direction=='LONG' else max(h.HIGH10,entry+1.5*h.ATR));risk=abs(entry-stop)
    blocked=(daily not in (direction,'NEUTRAL') or bias not in (direction,'NEUTRAL') or h.ADX<15 or dist>1.5 or risk<=0)
    signal='NO TRADE' if blocked else f'CONFIRMED {direction}' if score_>=85 else f'EARLY {direction}' if score_>=75 else f'WATCH {direction}' if score_>=65 else 'WAIT'
    grade='A+' if score_>=85 else 'A' if score_>=75 else 'B' if score_>=65 else 'C' if score_>=40 else '-'
    status='OPTIMAL' if dist<=.75 else 'AKZEPTABEL' if dist<=1 else 'SPÄT' if dist<=1.5 else 'KEIN EINSTIEG'
    sign=1 if direction=='LONG' else -1
    return {'Asset':meta['name'],'Ticker':meta['ticker'],'Signal':signal,'Direction':direction,'Grade':grade,'Score':score_,'Daily Trend':daily,'4h Bias':bias,'RSI':round(float(h.RSI),1),'ADX':round(float(h.ADX),1),'Entry Status':status,'Entry':round(entry,4),'Stop':round(stop,4),'TP1':round(entry+sign*2*risk,4),'TP2':round(entry+sign*3*risk,4),'Stop %':round(risk/entry*100,2),'CRV':2.0,'Haltedauer':'2-6 Wochen','Setup-Hinweis':f'Daily {daily}; 4h {bias}','Data Time':str(h4.index[-1])}
