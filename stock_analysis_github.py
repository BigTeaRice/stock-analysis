#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis – GitHub Actions 完整版
自動抓取數據 → 技術指標 → 繪製 K 線＋成交量 → 輸出到 docs/
適用於 yfinance（國際股）與 AkShare（A 股），後備模擬數據
作者：You
"""

import os
import warnings
from datetime import datetime, timedelta

import akshare as ak
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

###############################################################################
# 1. 組態區（GitHub Actions 友好）
###############################################################################
OUTPUT_DIR = "docs"                     # 圖表輸出根目錄
PERIOD = "3mo"                          # 默認 3 個月
TEST_STOCKS = [                         # 三組測試標的
    {"symbol": "AAPL", "source": "yfinance"},
    {"symbol": "000001", "source": "akshare"},
    {"symbol": "TEST", "source": "simulated"},
]


###############################################################################
# 2. 工具函式
###############################################################################
def ensure_dirs():
    """確保輸出目錄存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_path(symbol: str, idx: int) -> str:
    """回傳完整 png 路徑"""
    return os.path.join(OUTPUT_DIR, f"stock_{symbol}_{idx}.png")


###############################################################################
# 3. 資料源邏輯
###############################################################################
class StockAnalyzer:
    def __init__(self):
        self.data_source = "yfinance"
        self.cache = {}

    # ---------- 對外接口 ----------
    def set_data_source(self, source: str):
        self.data_source = source

    def fetch(self, symbol: str, period: str = PERIOD):
        print(f"🔍  Fetch {symbol} from {self.data_source}  ({period})")
        try:
            if self.data_source == "yfinance":
                return self._yfinance(symbol, period)
            if self.data_source == "akshare":
                return self._akshare(symbol, period)
            return self._simulate(symbol, period)
        except Exception as e:
            print(f"❌  Error: {e}  → fallback to simulated")
            return self._simulate(symbol, period)

    # ---------- yfinance ----------
    def _yfinance(self, symbol: str, period: str):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            # 自動加後綴再試
            for sfx in (".SI", ".HK"):
                t = yf.Ticker(symbol + sfx)
                h = t.history(period=period)
                if not h.empty:
                    symbol += sfx
                    hist = h
                    break
            else:
                raise ValueError("No data from yfinance")
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return {
            "symbol": symbol,
            "data": hist[["date", "open", "high", "low", "close", "volume"]],
            "info": ticker.info or {},
            "source": "yfinance",
        }

    # ---------- AkShare ----------
    def _akshare(self, symbol: str, period: str):
        # 簡易轉換 A 股代碼
        if symbol.startswith("6"):
            ak_sym = f"sh{symbol}"
        elif symbol.startswith(("0", "3")):
            ak_sym = f"sz{symbol}"
        else:
            ak_sym = symbol

        end = datetime.now()
        start = end - pd.Timedelta(days={"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 90))

        df = ak.stock_zh_a_hist(symbol=ak_sym[2:], period="daily",
                                start_date=start.strftime("%Y%m%d"),
                                end_date=end.strftime("%Y%m%d"), adjust="")
        if df.empty:
            raise ValueError("No data from AkShare")
        df = df.rename(columns={
            "日期": "date", "開盤": "open", "最高": "high",
            "最低": "low", "收盤": "close", "成交量": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        return {
            "symbol": symbol,
            "data": df[["date", "open", "high", "low", "close", "volume"]],
            "info": {"currency": "CNY", "exchange": "SSE/SZSE"},
            "source": "akshare",
        }

    # ---------- 模擬數據 ----------
    def _simulate(self, symbol: str, period: str):
        print("📊  Using simulated data …")
        days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}.get(period, 63)
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq="B")  # 僅工作日
        base = 100 + np.random.rand() * 50
        ret = np.random.randn(len(dates)) * 0.02
        prices = base * np.exp(np.cumsum(ret))
        ohlc = []
        for i, dt in enumerate(dates):
            close = prices[i]
            open_p = close * (1 + np.random.randn() * 0.01)
            high = max(open_p, close) * (1 + abs(np.random.randn()) * 0.015)
            low = min(open_p, close) * (1 - abs(np.random.randn()) * 0.015)
            vol = np.random.randint(1_000_000, 10_000_000)
            ohlc.append({"date": dt, "open": open_p, "high": high,
                         "low": low, "close": close, "volume": vol})
        df = pd.DataFrame(ohlc)
        return {
            "symbol": symbol,
            "data": df,
            "info": {"currency": "USD", "exchange": "SIM"},
            "source": "simulated",
        }

    # ---------- 技術指標 ----------
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        # MA
        df["MA5"] = close.rolling(5).mean()
        df["MA10"] = close.rolling(10).mean()
        df["MA20"] = close.rolling(20).mean()
        # RSI
        df["RSI"] = self._rsi(close.values, 14)
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        # 布林帶
        df["BB_Middle"] = df["MA20"]
        std20 = close.rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * std20
        df["BB_Lower"] = df["BB_Middle"] - 2 * std20
        # 成交量均線
        df["Volume_MA20"] = df["volume"].rolling(20).mean()
        return df

    @staticmethod
    def _rsi(prices, n=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.zeros_like(prices, dtype=float)
        avg_loss = np.zeros_like(prices, dtype=float)
        avg_gain[n] = gains[:n].mean()
        avg_loss[n] = losses[:n].mean()
        for i in range(n + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gains[i - 1]) / n
            avg_loss[i] = (avg_loss[i - 1] * (n - 1) + losses[i - 1]) / n
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi[:n] = 50
        return rsi

    # ---------- 繪圖 ----------
    def plot(self, stock_data: dict, indicators_df: pd.DataFrame, save_path: str):
        plt.style.use("seaborn-v0_8")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                       gridspec_kw={"height_ratios": [3, 1]})
        df = stock_data["data"]

        # K 線
        for _, row in df.iterrows():
            d, op, hi, lo, cl = row["date"], row["open"], row["high"], row["low"], row["close"]
            color = "red" if cl > op else "green"
            ax1.vlines(d, lo, hi, color=color, lw=1)
            ax1.vlines(d, min(op, cl), max(op, cl), color=color, lw=6)

        # MA
        ax1.plot(df["date"], indicators_df["MA5"], label="MA5", color="crimson", lw=1.2)
        ax1.plot(df["date"], indicators_df["MA20"], label="MA20", color="black", lw=1.2)

        ax1.set_title(f"{stock_data['symbol']}  K-line", fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # 成交量
        colors = ["red" if df["close"].iloc[i] >= df["open"].iloc[i] else "green"
                  for i in range(len(df))]
        ax2.bar(df["date"], df["volume"], color=colors, alpha=0.7, width=0.8)
        ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾  Chart saved → {save_path}")
        plt.close(fig)

    # ---------- 簡易報告 ----------
    def report(self, stock_data: dict, indicators_df: pd.DataFrame) -> dict:
        latest = stock_data["data"].iloc[-1]
        prev = stock_data["data"].iloc[-2] if len(stock_data["data"]) > 1 else latest
        chg = latest["close"] - prev["close"]
        chg_pct = chg / prev["close"] * 100
        return {
            "symbol": stock_data["symbol"],
            "current": latest["close"],
            "change": chg,
            "change_pct": chg_pct,
            "volume": latest["volume"],
            "rsi": indicators_df["RSI"].iloc[-1],
            "ma5": indicators_df["MA5"].iloc[-1],
            "ma20": indicators_df["MA20"].iloc[-1],
            "source": stock_data["source"],
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


###############################################################################
# 4. 主程式
###############################################################################
def main():
    ensure_dirs()
    analyzer = StockAnalyzer()
    for idx, cfg in enumerate(TEST_STOCKS, 1):
        symbol, src = cfg["symbol"], cfg["source"]
        print(f"\n{'='*60}")
        print(f"Run {idx}  {symbol}  ({src})")
        print('='*60)
        analyzer.set_data_source(src)
        data = analyzer.fetch(symbol, PERIOD)
        if data is None:
            continue
        ind = analyzer.indicators(data["data"].copy())
        rep = analyzer.report(data, ind)
        analyzer.plot(data, ind, save_path(symbol, idx))
        # 終端小報表
        print(f"📊  {rep['symbol']}  {rep['current']:.2f}  "
              f"({rep['change_pct']:+.2f}%)  RSI={rep['rsi']:.1f}")
        print(f"⏰  Updated at {rep['ts']}")


if __name__ == "__main__":
    print("🚀  Stock Analysis – GitHub Actions Edition")
    main()
