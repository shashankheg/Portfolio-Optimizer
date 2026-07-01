import gradio as gr
import json
import os
from datetime import datetime
from src.pipeline import run_pipeline, STOCKS, CONFIG



# ── CACHED RESULTS ────────────────────────────────────────────────────────────
cached_results = None

def load_cached_results():
    """Load last pipeline results if available."""
    path = "artifacts/pipeline_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_and_display(
    selected_stocks: list,
    strategy: str,
    max_weight: float,
    horizon_days: int
):
    
    """Run pipeline and return results for Gradio display."""
    global cached_results

    if not selected_stocks:
        return "⚠️ Please select at least 3 stocks.", "", "", ""
    
    config = {
        "horizon_days": int(horizon_days),
        "strategy":     strategy,
        "max_weight":   max_weight / 100,
        "min_weight":   0.02
    }

    try:
        results = run_pipeline(
            stocks=selected_stocks,
            config=config,
            save_results=True,
            send_email=False  # Don't send email from UI
        )
        cached_results = results
        return format_outputs(results)
    except Exception as e:
        return f"❌ Pipeline failed: {e}", "", "", ""
    

def format_outputs(results):
    """Format pipeline results for display."""

    # Portfolio table
    portfolio  = results.get("portfolio", {})
    forecasts  = results.get("forecasts", {})
    sentiment  = results.get("sentiment", {})
    metrics    = results.get("metrics", {})
    explanation = results.get("explanation", "Not available")


    # Portfolio allocation text
    portfolio_text = "📊 PORTFOLIO ALLOCATION\n"
    portfolio_text += "─" * 45 + "\n"
    portfolio_text += f"{'Stock':<8} {'Weight':>8} {'Forecast':>10} {'Sentiment':>12}\n"
    portfolio_text += "─" * 45 + "\n"

    for stock, weight in sorted(portfolio.items(),
                                key=lambda x: x[1], reverse=True):
        ret = forecasts.get(stock, {}).get("return", 0)
        sig = sentiment.get(stock, {}).get("signal", "neutral")
        icon = "🟢" if sig == "bullish" else "🔴" if sig == "bearish" else "⚪"
        bar  = "█" * int(weight * 40)
        portfolio_text += (f"{stock:<8} {weight:>7.1%} {ret:>+10.2%} "
                           f"{icon} {sig:<10}\n")
        portfolio_text += f"         {bar}\n"


# Metrics text
    metrics_text = "📈 PORTFOLIO METRICS\n"
    metrics_text += "─" * 35 + "\n"
    metrics_text += f"Expected Return:  {metrics.get('expected_return', 0):>+8.2%}\n"
    metrics_text += f"Volatility:       {metrics.get('volatility', 0):>8.2%}\n"
    metrics_text += f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>8.2f}\n"
    metrics_text += f"Max Drawdown:     {metrics.get('max_drawdown', 0):>8.2%}\n"
    metrics_text += f"Beta:             {metrics.get('beta', 0):>8.2f}\n"

# Sentiment text
    sentiment_text = "🤖 SENTIMENT SIGNALS\n"
    sentiment_text += "─" * 35 + "\n"
    for stock, sent in sentiment.items():
        score = sent.get("sentiment_score", 0)
        sig   = sent.get("signal", "neutral")
        conf  = sent.get("confidence", 0)
        icon  = "🟢" if sig == "bullish" else "🔴" if sig == "bearish" else "⚪"
        sentiment_text += (f"{stock:<6} {icon} {sig:<8} "
                           f"score={score:+.2f} conf={conf:.0%}\n")

    return portfolio_text, metrics_text, sentiment_text, explanation

custom_css = """
:root {
    --primary: #0D9488;
    --primary-light: #5EEAD4;
    --border: #ccfbf1;
    --surface: #f0fdfa;
    --radius: 16px;
    --shadow: 0 4px 24px rgba(13,148,136,0.10);
    }
body, .gradio-container {
    background: linear-gradient(135deg, #f0fdfa, #e6fffa) !important;
    font-family: 'Inter', sans-serif !important;
}
.app-header {
    background: linear-gradient(135deg, #0D9488, #14B8A6);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
}
.card {
    background: white !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow) !important;
}
button.primary {
    background: linear-gradient(135deg, #0D9488, #14B8A6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
"""

ALL_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    "META", "AMZN", "NFLX", "NOW", "AMD",
    "INTC", "CRM", "ORCL", "ADBE", "PYPL"
]


def create_app():
    with gr.Blocks(title="Portfolio Optimizer", css=custom_css) as app:

        gr.HTML("""
        <div class="app-header">
            <div style="font-size:40px">📈</div>
            <h1 style="font-size:28px;font-weight:800;margin:8px 0">
                AI Portfolio Optimizer
            </h1>
            <p style="opacity:0.9;margin:0">
                Time Series Forecasting · GenAI Sentiment · Portfolio Optimization
            </p>
        </div>
        """)
        with gr.Row():
            # ── LEFT PANEL — Controls ──────────────────────────────────────
            with gr.Column(scale=1, elem_classes="card"):
                gr.HTML("<h3 style='color:#134e4a'>⚙️ Configuration</h3>")

                selected_stocks = gr.CheckboxGroup(
                    label="Select Stocks",
                    choices=ALL_STOCKS,
                    value=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                )

                strategy = gr.Dropdown(
                    label="Optimization Strategy",
                    choices=[
                        "max_sharpe",
                        "min_variance",
                        "max_return",
                        "risk_parity"
                    ],
                    value="max_sharpe"
                )

                max_weight = gr.Slider(
                    label="Max Weight per Stock (%)",
                    minimum=10,
                    maximum=60,
                    value=40,
                    step=5
                )

                horizon = gr.Slider(
                    label="Forecast Horizon (days)",
                    minimum=7,
                    maximum=90,
                    value=30,
                    step=7
                )

                run_btn = gr.Button(
                    "🚀 Run Portfolio Optimization",
                    variant="primary"
                )

                gr.HTML("""
                <div style="margin-top:12px;padding:10px;background:#f0fdfa;
                            border-radius:8px;font-size:12px;color:#6b7280">
                    ⏱️ Takes ~30-60 seconds to complete<br>
                    🤖 Uses Groq LLaMA 3 for sentiment analysis<br>
                    📊 Gradient Boosting for return forecasting
                </div>
                """)


                # ── RIGHT PANEL — Results ──────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Portfolio"):
                        portfolio_out = gr.Textbox(
                            label="Portfolio Allocation",
                            lines=20,
                            interactive=False,
                            elem_classes="card"
                        )

                    with gr.Tab("📈 Metrics"):
                        metrics_out = gr.Textbox(
                            label="Performance Metrics",
                            lines=10,
                            interactive=False,
                            elem_classes="card"
                        )

                    with gr.Tab("🤖 Sentiment"):
                        sentiment_out = gr.Textbox(
                            label="GenAI Sentiment Signals",
                            lines=12,
                            interactive=False,
                            elem_classes="card"
                        )

                    with gr.Tab("💬 AI Analysis"):
                        explanation_out = gr.Textbox(
                            label="LLM Portfolio Explanation",
                            lines=12,
                            interactive=False,
                            elem_classes="card"
                        )

            run_btn.click(
            fn=run_and_display,
            inputs=[selected_stocks, strategy, max_weight, horizon],
            outputs=[portfolio_out, metrics_out, sentiment_out, explanation_out]
        )
            
        gr.HTML("""
        <div style="text-align:center;padding:1.5rem;color:#6b7280;font-size:12px">
            ⚠️ For informational purposes only. Not financial advice.<br>
            Built with LangChain · LangGraph · Groq · Gradio
        </div>
        """)

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

