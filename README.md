---
title: AI Portfolio Optimizer
emoji: 📈
colorFrom: green
colorTo: teal
sdk: docker
pinned: false
---

# 📈 AI Portfolio Optimizer

> Time series forecasting + GenAI sentiment analysis + portfolio optimization in one pipeline.

---

## 🔧 Components

### 📥 Data Loader
Fetches 3 years of historical OHLCV stock data using yfinance.
Computes technical indicators including MA, RSI, MACD, and Bollinger Bands.
Builds a clean returns matrix ready for modeling.

### 📈 Time Series Forecasting
Uses Gradient Boosting to predict 30-day forward returns for each stock.
Incorporates trend, seasonality, momentum, and lagged features.
Outputs predicted return, volatility, confidence, and direction per stock.

### 🤖 GenAI Features
Uses Groq LLaMA 3 to analyze market sentiment for each stock.
Fetches macroeconomic signals including interest rates, inflation, and market outlook.
Converts LLM text outputs into numeric features that feed into the optimizer.

### ⚖️ Portfolio Optimization
Finds the mathematically optimal allocation across all selected stocks.
Supports 4 strategies — Max Sharpe, Min Variance, Max Return, Risk Parity.
Enforces constraints like max 40% per stock and minimum 2% per position.

### 💬 LLM Explanation
Uses LLaMA 3.3 70B to generate a human-readable daily portfolio briefing.
Explains top conviction positions, key risks, and one actionable insight.
Makes the portfolio decisions transparent and easy to understand.

### 📧 Daily Email Report
Sends a rich HTML email every weekday at 9 AM via SendGrid.
Includes portfolio allocation, metrics, sentiment signals, and AI analysis.
Scheduled automatically via GitHub Actions — no manual intervention needed.

### 🖥️ Gradio UI
Interactive web interface to run the pipeline with custom stock selections.
Supports configurable optimization strategy, max weight, and forecast horizon.
Displays results across 4 tabs — Portfolio, Metrics, Sentiment, AI Analysis.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq LLaMA 3.3 70B + LLaMA 3.1 8B |
| **Forecasting** | Gradient Boosting (scikit-learn) |
| **Optimization** | SciPy SLSQP solver |
| **UI** | Gradio |
| **Email** | SendGrid |
| **Scheduling** | GitHub Actions |
| **Deployment** | Docker + Hugging Face Spaces |

---

## 🚀 Local Setup

```bash
git clone https://github.com/shashankheg/Portfolio-Optimizer.git
cd Portfolio-Optimizer
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
# Add keys to .env file
python -m src.app
```

---

## ⚠️ Disclaimer
For informational purposes only. Not financial advice.