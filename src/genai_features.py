import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm import get_llm, get_fast_llm


load_dotenv()

# ── PROMPTS ──────────────────────────────────────────────────────────────────

SENTIMENT_PROMPT = PromptTemplate(
    input_variables=["stock", "sector", "recent_performance"],
    template="""You are a financial analyst specializing in stock sentiment analysis.

Analyze the market sentiment for {stock} stock.
Sector: {sector}
Recent Performance: {recent_performance}

Based on general market knowledge, current AI/tech trends, and typical 
market dynamics for this stock, provide a sentiment analysis.

Return ONLY a JSON object, no extra text:
{{
    "sentiment_score": <float between -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float between 0.0 and 1.0>,
    "signal": "<bullish|bearish|neutral>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "risk_factors": ["<risk1>", "<risk2>"]
}}"""
)

MACRO_PROMPT = PromptTemplate(
    input_variables=["current_date"],
    template="""You are a macroeconomic analyst.

Based on your knowledge of global economic conditions as of {current_date},
analyze the current macroeconomic environment for equity investing.

Return ONLY a JSON object, no extra text:
{{
    "interest_rate_impact": "<positive|negative|neutral>",
    "inflation_signal": "<high|moderate|low>",
    "market_outlook": "<bullish|bearish|neutral>",
    "gdp_trend": "<expanding|contracting|stable>",
    "sectors_to_overweight": ["<sector1>", "<sector2>", "<sector3>"],
    "sectors_to_underweight": ["<sector1>", "<sector2>"],
    "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
    "overall_score": <float between -1.0 (very bearish) to 1.0 (very bullish)>
}}"""
)

PORTFOLIO_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["portfolio", "metrics", "forecasts", "sentiment"],
    template="""You are a senior portfolio manager providing a daily briefing.

Portfolio Weights: {portfolio}
Performance Metrics: {metrics}
Stock Forecasts: {forecasts}
Sentiment Signals: {sentiment}

Provide a concise professional portfolio briefing covering:
1. Overall market outlook
2. Top 2 conviction positions and why
3. Main risks to monitor today
4. One specific actionable insight

Keep it under 200 words. Be specific with numbers."""
)

# ── STOCK METADATA ────────────────────────────────────────────────────────────

STOCK_SECTORS = {
    "AAPL":  "Technology - Consumer Electronics",
    "MSFT":  "Technology - Cloud & Software",
    "GOOGL": "Technology - Digital Advertising & AI",
    "TSLA":  "Consumer Discretionary - Electric Vehicles",
    "AMZN":  "Technology - E-commerce & Cloud",
    "NVDA":  "Technology - Semiconductors & AI",
    "META":  "Technology - Social Media & AR/VR",
    "NFLX":  "Communication Services - Streaming",
    "NOW":   "Technology - Enterprise Software",
}


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def analyze_sentiment(
    stocks: List[str],
    forecasts: Dict = None
) -> Dict[str, Dict]:
    """
    Use LLM to analyze sentiment for each stock.

    Args:
        stocks: List of stock tickers
        forecasts: Optional forecast dict for context

    Returns:
        Dict of {ticker: sentiment_dict}
    """
    print("\n🤖 Analyzing sentiment with LLM...")

    llm   = get_fast_llm()
    chain = SENTIMENT_PROMPT | llm | StrOutputParser()
    sentiment = {}

    for stock in stocks:
        try:
            sector = STOCK_SECTORS.get(stock, "Unknown")

            # Build recent performance context
            if forecasts and stock in forecasts:
                f = forecasts[stock]
                recent = (f"Forecast return: {f['return']:+.2%}, "
                          f"Volatility: {f['volatility']:.2%}, "
                          f"Direction: {f['direction']}")
            else:
                recent = "No recent forecast available"

            result = chain.invoke({
                "stock":              stock,
                "sector":             sector,
                "recent_performance": recent
            })

            # Parse JSON response
            start = result.find("{")
            end   = result.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(result[start:end])
                sentiment[stock] = parsed
                print(f"   ✅ {stock}: score={parsed['sentiment_score']:+.2f} "
                      f"| signal={parsed['signal']} "
                      f"| conf={parsed['confidence']:.2f}")
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            print(f"   ❌ {stock} sentiment failed: {e}")
            sentiment[stock] = _fallback_sentiment(stock)

    return sentiment


def get_macro_signals() -> Dict:
    """
    Use LLM to analyze current macroeconomic conditions.

    Returns:
        Dict with macro signals and outlook
    """
    print("\n🌍 Fetching macro economic signals...")

    from datetime import datetime
    llm   = get_llm()
    chain = MACRO_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "current_date": datetime.now().strftime("%B %Y")
        })

        start = result.find("{")
        end   = result.rfind("}") + 1
        if start != -1 and end > start:
            macro = json.loads(result[start:end])
            print(f"   ✅ Market outlook: {macro['market_outlook']}")
            print(f"   ✅ Interest rate impact: {macro['interest_rate_impact']}")
            print(f"   ✅ Inflation signal: {macro['inflation_signal']}")
            print(f"   ✅ Sectors to overweight: "
                  f"{macro['sectors_to_overweight']}")
            return macro

    except Exception as e:
        print(f"   ❌ Macro signals failed: {e}")

    return _fallback_macro()


def encode_genai_features(
    sentiment: Dict[str, Dict],
    macro: Dict,
    stocks: List[str]
) -> Dict[str, Dict]:
    """
    Convert GenAI outputs into numeric features for the model.

    Args:
        sentiment: Sentiment dict from analyze_sentiment()
        macro: Macro dict from get_macro_signals()
        stocks: List of stock tickers

    Returns:
        Dict of {ticker: numeric_features}
    """
    signal_map  = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    impact_map  = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    inflation_map = {"low": 1.0, "moderate": 0.0, "high": -1.0}

    macro_score = macro.get("overall_score", 0.0)
    rate_impact = impact_map.get(macro.get("interest_rate_impact", "neutral"), 0)
    infl_signal = inflation_map.get(macro.get("inflation_signal", "moderate"), 0)

    features = {}
    for stock in stocks:
        sent = sentiment.get(stock, _fallback_sentiment(stock))
        features[stock] = {
            "sentiment_score":    float(sent.get("sentiment_score", 0)),
            "sentiment_signal":   signal_map.get(sent.get("signal", "neutral"), 0),
            "sentiment_confidence": float(sent.get("confidence", 0.5)),
            "macro_score":        float(macro_score),
            "rate_impact":        float(rate_impact),
            "inflation_signal":   float(infl_signal),
        }

    return features


def explain_portfolio(
    portfolio: Dict,
    metrics: Dict,
    forecasts: Dict,
    sentiment: Dict
) -> str:
    """
    Use LLM to generate a human-readable portfolio explanation.

    Returns:
        String explanation from LLM
    """
    print("\n💬 Generating portfolio explanation...")

    llm   = get_llm()
    chain = PORTFOLIO_EXPLAIN_PROMPT | llm | StrOutputParser()

    try:

        # Filter metrics to only numeric top-level values
        clean_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        explanation = chain.invoke({
            "portfolio": json.dumps(
                {k: f"{v:.1%}" for k, v in portfolio.items()}, indent=2),
            "metrics":   json.dumps(
                {k: f"{v:.3f}" for k, v in clean_metrics.items()}, indent=2),
            "forecasts": json.dumps(
                {k: f"{v['return']:+.2%}" for k, v in forecasts.items()}, indent=2),
            "sentiment": json.dumps(
                {k: v.get("signal", "neutral") for k, v in sentiment.items()},
                indent=2)
        })
        print("   ✅ Explanation generated")
        return explanation

    except Exception as e:
        print(f"   ❌ Explanation failed: {e}")
        return "Portfolio explanation unavailable."


# ── FALLBACKS ─────────────────────────────────────────────────────────────────

def _fallback_sentiment(stock: str) -> Dict:
    return {
        "sentiment_score": 0.0,
        "confidence":      0.5,
        "signal":          "neutral",
        "key_factors":     ["insufficient data"],
        "risk_factors":    ["unknown"]
    }


def _fallback_macro() -> Dict:
    return {
        "interest_rate_impact":   "neutral",
        "inflation_signal":       "moderate",
        "market_outlook":         "neutral",
        "gdp_trend":              "stable",
        "sectors_to_overweight":  ["technology"],
        "sectors_to_underweight": ["utilities"],
        "key_risks":              ["market uncertainty"],
        "overall_score":          0.0
    }


if __name__ == "__main__":
    from src.data_loader import get_all_data
    from src.time_series import forecast_returns

    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    # Get forecasts for context
    data      = get_all_data(stocks)
    forecasts = forecast_returns(data["enriched"])

    # Get GenAI features
    sentiment = analyze_sentiment(stocks, forecasts)
    macro     = get_macro_signals()
    features  = encode_genai_features(sentiment, macro, stocks)

    print("\n📊 Encoded GenAI Features:")
    for stock, feat in features.items():
        print(f"  {stock}: sentiment={feat['sentiment_score']:+.2f} "
              f"| macro={feat['macro_score']:+.2f} "
              f"| rate={feat['rate_impact']:+.1f}")
