import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import get_all_data
from src.time_series import forecast_returns
from src.genai_features import analyze_sentiment, get_macro_signals, \
    encode_genai_features, explain_portfolio
from src.portfolio import optimize_portfolio, compare_strategies
from src.email_report import send_daily_report

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

STOCKS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"]

CONFIG = {
    "horizon_days": 30,        # forecast horizon
    "strategy":     "max_sharpe",  # optimization strategy
    "max_weight":   0.40,      # max 40% per stock
    "min_weight":   0.02,      # min 2% per stock
}


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(
    stocks: list = STOCKS,
    config: dict = CONFIG,
    save_results: bool = True
) -> dict:
    """
    Run the complete portfolio optimization pipeline.

    Steps:
    1. Fetch stock data
    2. Compute technical indicators
    3. Forecast returns (ML)
    4. Get GenAI sentiment + macro signals
    5. Optimize portfolio
    6. Generate LLM explanation
    7. Save results

    Returns:
        Dict with complete pipeline results
    """
    start_time = datetime.now()
    print("\n" + "🚀 " * 20)
    print("   PORTFOLIO OPTIMIZER — DAILY PIPELINE")
    print(f"   {start_time.strftime('%A, %B %d, %Y — %H:%M:%S')}")
    print("🚀 " * 20)

    results = {
        "timestamp": start_time.isoformat(),
        "stocks":    stocks,
        "config":    config
    }

    # ── STAGE 1: DATA LOADING ─────────────────────────────────────────────
    print("\n📥 STAGE 1: Loading market data...")
    try:
        data = get_all_data(stocks)
        results["available_stocks"] = data["stocks"]
        print(f"✅ Stage 1 complete — {len(data['stocks'])} stocks loaded")
    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        raise

    # ── STAGE 2: TIME SERIES FORECASTING ──────────────────────────────────
    print("\n📈 STAGE 2: Forecasting returns...")
    try:
        forecasts = forecast_returns(
            data["enriched"],
            horizon_days=config["horizon_days"]
        )
        results["forecasts"] = forecasts
        print(f"✅ Stage 2 complete — {len(forecasts)} stocks forecasted")
    except Exception as e:
        print(f"❌ Stage 2 failed: {e}")
        raise

    # ── STAGE 3: GENAI FEATURES ───────────────────────────────────────────
    print("\n🤖 STAGE 3: Getting GenAI signals...")
    try:
        sentiment      = analyze_sentiment(data["stocks"], forecasts)
        macro          = get_macro_signals()
        genai_features = encode_genai_features(
            sentiment, macro, data["stocks"]
        )
        results["sentiment"]      = sentiment
        results["macro"]          = macro
        results["genai_features"] = genai_features
        print(f"✅ Stage 3 complete — sentiment + macro signals ready")
    except Exception as e:
        print(f"⚠️  Stage 3 failed: {e} — continuing without GenAI features")
        sentiment      = {}
        macro          = {}
        genai_features = {}

    # ── STAGE 4: PORTFOLIO OPTIMIZATION ───────────────────────────────────
    print("\n⚖️  STAGE 4: Optimizing portfolio...")
    try:
        portfolio, metrics = optimize_portfolio(
            data["returns"],
            forecasts,
            genai_features,
            strategy=config["strategy"],
            max_weight=config["max_weight"],
            min_weight=config["min_weight"]
        )
        results["portfolio"] = portfolio
        results["metrics"]   = metrics
        print(f"✅ Stage 4 complete — portfolio optimized")
    except Exception as e:
        print(f"❌ Stage 4 failed: {e}")
        raise

    # ── STAGE 5: LLM EXPLANATION ──────────────────────────────────────────
    print("\n💬 STAGE 5: Generating AI explanation...")
    try:
        explanation = explain_portfolio(
            portfolio, metrics, forecasts, sentiment
        )
        results["explanation"] = explanation
        print(f"✅ Stage 5 complete — explanation generated")
    except Exception as e:
        print(f"⚠️  Stage 5 failed: {e}")
        results["explanation"] = "Explanation unavailable."

    
        # ── STAGE 6: SEND EMAIL REPORT ────────────────────────────────────────
    print("\n📧 STAGE 6: Sending email report...")
    try:
        email_sent = send_daily_report(results)
        results["email_sent"] = email_sent
    except Exception as e:
        print(f"⚠️  Stage 6 failed: {e}")
        results["email_sent"] = False

    # ── STAGE 7: SAVE RESULTS ─────────────────────────────────────────────
    if save_results:
        print("\n💾 STAGE 7: Saving results...")
        try:
            os.makedirs("artifacts", exist_ok=True)
            output_path = "artifacts/pipeline_results.json"

            # Make results JSON serializable
            serializable = make_serializable(results)

            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)

            print(f"✅ Results saved to {output_path}")
        except Exception as e:
            print(f"⚠️  Save failed: {e}")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'='*55}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed}s")
    print(f"{'='*55}")
    print(f"  Expected Return: {metrics['expected_return']:+.2%}")
    print(f"  Volatility:      {metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"\n  Portfolio Allocation:")
    for stock, weight in sorted(portfolio.items(),
                                key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"    {stock:<6} {weight:>6.1%}  {bar}")
    print(f"{'='*55}\n")

    return results


def make_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    results = run_pipeline()