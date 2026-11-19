from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

MIN_RR = 2.0

def _download(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(how='all')

def _confirmed(data, interval):
    if data.empty: return data
    now = pd.Timestamp.now(tz='UTC')
    idx = data.index
    last = idx[-1]
    if getattr(last, 'tzinfo', None) is None: last = last.tz_localize('UTC')
    else: last = last.tz_convert('UTC')
    duration = pd.Timedelta(hours=1) if interval == '1h' else pd.Timedelta(days=1)
    return data.iloc[:-1] if now < last + duration else data

def _resample_4h(data):
    rules = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    cols = {k:v for k,v in rules.items() if k in data.columns}
    return data.resample('4h', origin='start_day').agg(cols).dropna(subset=['Open','High','Low','Close'])

def _indicators(df):
    d = df.copy()
    for n in (20,50,200): d[f'EMA{n}'] = EMAIndicator(d['Close'], window=n).ema_indicator()
    d['RSI'] = RSIIndicator(d['Close'], window=14).rsi()
    macd = MACD(d['Close']); d['MACD_H'] = macd.macd_diff()
    d['ADX'] = ADXIndicator(d['High'],d['Low'],d['Close'],window=14).adx()
    d['ATR'] = AverageTrueRange(d['High'],d['Low'],d['Close'],window=14).average_true_range()
    if 'Volume' in d and d['Volume'].notna().any(): d['RelVol'] = d['Volume']/d['Volume'].rolling(20).mean()
    else: d['RelVol'] = 1.0
    d['SwingLow'] = d['Low'].rolling(10).min(); d['SwingHigh'] = d['High'].rolling(10).max()
    return d.dropna()

def _score(side, h, d):
    long = side == 'LONG'; score = 0; reasons=[]
    tests=[
        (d.Close>d.EMA200 if long else d.Close<d.EMA200,15,'Daily vs EMA200'),
        (d.EMA20>d.EMA50 if long else d.EMA20<d.EMA50,10,'Daily EMA20/50'),
        (d.MACD_H>0 if long else d.MACD_H<0,10,'Daily MACD'),
        (h.Close>h.EMA20 and h.Close>h.EMA50 if long else h.Close<h.EMA20 and h.Close<h.EMA50,15,'4h Trend'),
        (h.EMA20>h.EMA50 if long else h.EMA20<h.EMA50,10,'4h EMA20/50'),
        (52<=h.RSI<=68 if long else 32<=h.RSI<=48,10,'4h RSI'),
        (h.MACD_H>0 if long else h.MACD_H<0,10,'4h MACD'),
        (h.ADX>=20,10,'ADX'),
        (abs(h.Close-h.EMA20)<=h.ATR,5,'Entry-Distanz'),
        (h.RelVol>=1,5,'Relatives Volumen')]
    for ok,points,label in tests:
        if bool(ok): score += points
        reasons.append(f'{label}: {"OK" if ok else "offen"}')
    return score,reasons

def scan_swing_asset(meta):
    ticker=meta['ticker']; raw1=_confirmed(_download(ticker,'730d','1h'),'1h'); daily=_confirmed(_download(ticker,'3y','1d'),'1d')
    if len(raw1)<300 or len(daily)<220: raise ValueError('Zu wenig bestätigte Daten')
    h4=_indicators(_resample_4h(raw1)); dd=_indicators(daily)
    if h4.empty or dd.empty: raise ValueError('Indikatoren unvollständig')
    h,d=h4.iloc[-1],dd.iloc[-1]
    ls,lr=_score('LONG',h,d); ss,sr=_score('SHORT',h,d)
    direction='LONG' if ls>=ss else 'SHORT'; score=max(ls,ss); reasons=lr if direction=='LONG' else sr
    daily_dir='LONG' if d.Close>d.EMA200 and d.EMA20>d.EMA50 else 'SHORT' if d.Close<d.EMA200 and d.EMA20<d.EMA50 else 'NEUTRAL'
    h4_dir='LONG' if h.Close>h.EMA50 and h.EMA20>h.EMA50 else 'SHORT' if h.Close<h.EMA50 and h.EMA20<h.EMA50 else 'NEUTRAL'
    distance=abs(h.Close-h.EMA20)/h.ATR if h.ATR else 99
    conflict=daily_dir not in (direction,'NEUTRAL') or h4_dir not in (direction,'NEUTRAL')
    entry=float(h.Close)
    stop=float(min(h.SwingLow,entry-1.5*h.ATR) if direction=='LONG' else max(h.SwingHigh,entry+1.5*h.ATR))
    risk=abs(entry-stop); tp1=entry+(2*risk if direction=='LONG' else -2*risk); tp2=entry+(3*risk if direction=='LONG' else -3*risk)
    entry_status='OPTIMAL' if distance<=.75 else 'AKZEPTABEL' if distance<=1 else 'SPÄT' if distance<=1.5 else 'KEIN EINSTIEG'
    blocked=conflict or h.ADX<15 or distance>1.5 or risk<=0
    if blocked: signal='NO TRADE'
    elif score>=85: signal=f'CONFIRMED {direction}'
    elif score>=75: signal=f'EARLY {direction}'
    elif score>=65: signal=f'WATCH {direction}'
    else: signal='WAIT'
    grade='A+' if score>=85 else 'A' if score>=75 else 'B' if score>=65 else 'C' if score>=40 else '-'
    return {'Asset':meta['name'],'Ticker':ticker,'Signal':signal,'Direction':direction,'Grade':grade,'Score':int(score),
            'Daily Trend':daily_dir,'4h Bias':h4_dir,'RSI':round(float(h.RSI),1),'ADX':round(float(h.ADX),1),
            'Entry Status':entry_status,'Entry':round(entry,4),'Stop':round(stop,4),'TP1':round(tp1,4),'TP2':round(tp2,4),
            'Stop %':round(risk/entry*100,2),'CRV':MIN_RR,'Haltedauer':'2-6 Wochen',
            'Setup-Hinweis':' · '.join(reasons[:5]),'Data Time':str(h4.index[-1]),'Trading Mode':'SWING'}
