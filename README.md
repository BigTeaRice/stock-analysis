# 股票分析工具（Stock Analysis Tool）

自動化股票數據分析與可視化工具，支援 yfinance、AkShare 與模擬數據。

## 功能
- 自動抓取股票數據（A股、美股）
- 技術指標計算（MA、RSI、MACD、布林帶）
- K線圖與成交量圖表生成
- 靜態網頁可視化（HTML）

## 使用
- 修改 `stock_analysis_github.py` 中的測試案例
- 圖表會生成於 `docs/` 資料夾
- HTML 網頁位於 `docs/index.html`

## 部署
- Python 腳本每日自動運行（GitHub Actions）
- 靜態網頁部署於 GitHub Pages
