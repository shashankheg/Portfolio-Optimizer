import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize


# ── PORTFOLIO OPTIMIZATION ────────────────────────────────────────────────────

def compute_portfolio_metrics(
    weights: np.ndarray,
    returns: pd.DataFrame,
    forecasts: Dict,
    genai_features: Dict = None
) -> Dict:
    """
    Compute portfolio performance metrics.

    Args:
        weights: Array of portfolio weights
        returns: DataFrame of historical daily returns
        forecasts: Dict of stock forecasts
        genai_features: Optional GenAI encoded features

    Returns:
        Dict of portfolio metrics
    """
    stocks = list(returns.columns)

    # Covariance matrix (annualized)
    cov_matrix = returns.cov() * 252

    # Expected returns — blend forecast + historical + GenAI
    historical_returns = returns.mean() * 252

    expected_returns = {}
    for stock in stocks:
        hist_ret  = historical_returns.get(stock, 0)
        fore_ret  = forecasts.get(stock, {}).get("return", 0) * (252 / 30)

        # Blend historical (40%) + forecast (40%) + GenAI sentiment (20%)
        genai_boost = 0.0
        if genai_features and stock in genai_features:
            sentiment_score = genai_features[stock].get("sentiment_score", 0)
            macro_score     = genai_features[stock].get("macro_score", 0)
            genai_boost     = (sentiment_score * 0.03) + (macro_score * 0.02)

        expected_returns[stock] = (
            0.40 * hist_ret +
            0.40 * fore_ret +
            0.20 * genai_boost
        )

    exp_ret_array = np.array([expected_returns[s] for s in stocks])

    # Portfolio metrics
    port_return   = float(np.dot(weights, exp_ret_array))
    port_variance = float(np.dot(weights, np.dot(cov_matrix.values, weights)))
    port_vol      = float(np.sqrt(port_variance))
    sharpe        = float((port_return - 0.05) / (port_vol + 1e-8))  # rf = 5%

    # Max drawdown approximation
    weighted_returns = returns.dot(weights)
    cumulative       = (1 + weighted_returns).cumprod()
    rolling_max      = cumulative.cummax()
    drawdown         = (cumulative - rolling_max) / rolling_max
    max_drawdown     = float(drawdown.min())

    # Beta (vs equal-weighted market)
    market_returns = returns.mean(axis=1)
    port_returns   = returns.dot(weights)
    cov_with_mkt   = np.cov(port_returns, market_returns)[0][1]
    market_var     = market_returns.var()
    beta           = float(cov_with_mkt / (market_var + 1e-8))

    return {
        "expected_return": port_return,
        "volatility":      port_vol,
        "sharpe_ratio":    sharpe,
        "max_drawdown":    max_drawdown,
        "beta":            beta,
        "expected_returns_per_stock": expected_returns
    }


def optimize_portfolio(
    returns: pd.DataFrame,
    forecasts: Dict,
    genai_features: Dict = None,
    strategy: str = "max_sharpe",
    max_weight: float = 0.40,
    min_weight: float = 0.02
) -> Tuple[Dict, Dict]:
    """
    Optimize portfolio weights using scipy.

    Args:
        returns: DataFrame of daily returns
        forecasts: Stock forecasts from time series model
        genai_features: GenAI encoded features
        strategy: Optimization strategy
            - 'max_sharpe': Maximize Sharpe ratio
            - 'min_variance': Minimize portfolio variance
            - 'max_return': Maximize expected return
            - 'risk_parity': Equal risk contribution
        max_weight: Maximum weight per stock
        min_weight: Minimum weight per stock

    Returns:
        Tuple of (weights_dict, metrics_dict)
    """
    print(f"\n⚖️  Optimizing portfolio — strategy: {strategy}")

    stocks  = list(returns.columns)
    n       = len(stocks)
    cov_mat = returns.cov().values * 252

    # Build expected returns array
    historical_returns = (returns.mean() * 252).values
    forecast_returns   = np.array([
        forecasts.get(s, {}).get("return", 0) * (252 / 30)
        for s in stocks
    ])

    genai_boost = np.zeros(n)
    if genai_features:
        for i, stock in enumerate(stocks):
            if stock in genai_features:
                sent  = genai_features[stock].get("sentiment_score", 0)
                macro = genai_features[stock].get("macro_score", 0)
                genai_boost[i] = sent * 0.03 + macro * 0.02

    exp_returns = (
        0.40 * historical_returns +
        0.40 * forecast_returns   +
        0.20 * genai_boost
    )

    # === OBJECTIVE FUNCTIONS ===

    def neg_sharpe(w):
        ret = np.dot(w, exp_returns)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w)))
        return -(ret - 0.05) / (vol + 1e-8)

    def portfolio_variance(w):
        return np.dot(w, np.dot(cov_mat, w))

    def neg_return(w):
        return -np.dot(w, exp_returns)

    def risk_parity_objective(w):
        vol      = np.sqrt(np.dot(w, np.dot(cov_mat, w)))
        mrc      = np.dot(cov_mat, w) / (vol + 1e-8)
        rc       = w * mrc
        target   = vol / n
        return float(np.sum((rc - target) ** 2))

    objectives = {
        "max_sharpe":    neg_sharpe,
        "min_variance":  portfolio_variance,
        "max_return":    neg_return,
        "risk_parity":   risk_parity_objective
    }

    objective = objectives.get(strategy, neg_sharpe)

    # === CONSTRAINTS & BOUNDS ===
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = tuple((min_weight, max_weight) for _ in range(n))

    # Initial weights — equal weight
    w0 = np.array([1.0 / n] * n)

    # === SOLVE ===
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if not result.success:
        print(f"   ⚠️  Optimizer warning: {result.message}")
        print(f"   ↩️  Falling back to equal weights")
        weights = w0
    else:
        weights = result.x
        # Normalize to ensure sum = 1
        weights = weights / weights.sum()

    # Build weights dict
    weights_dict = {stock: float(w) for stock, w in zip(stocks, weights)}

    # Compute metrics
    metrics = compute_portfolio_metrics(
        weights, returns, forecasts, genai_features
    )

    # Print results
    print(f"\n   📊 Optimized Portfolio ({strategy}):")
    print(f"   {'Stock':<8} {'Weight':>8} {'Exp.Return':>12} {'Signal':>10}")
    print(f"   {'─'*42}")
    for stock, w in sorted(weights_dict.items(),
                           key=lambda x: x[1], reverse=True):
        exp_ret = metrics["expected_returns_per_stock"].get(stock, 0)
        signal  = (genai_features or {}).get(stock, {}).get("sentiment_signal", 0)
        sig_str = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        print(f"   {stock:<8} {w:>8.1%} {exp_ret:>+12.2%} {sig_str:>10}")

    print(f"\n   Expected Return: {metrics['expected_return']:+.2%}")
    print(f"   Volatility:      {metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"   Beta:            {metrics['beta']:.2f}")

    return weights_dict, metrics


def compare_strategies(
    returns: pd.DataFrame,
    forecasts: Dict,
    genai_features: Dict = None
) -> Dict:
    """
    Compare all optimization strategies side by side.

    Returns:
        Dict of {strategy: (weights, metrics)}
    """
    print("\n📊 Comparing all optimization strategies...")
    strategies = ["max_sharpe", "min_variance", "max_return", "risk_parity"]
    results    = {}

    for strategy in strategies:
        try:
            weights, metrics = optimize_portfolio(
                returns, forecasts, genai_features, strategy=strategy
            )
            results[strategy] = {"weights": weights, "metrics": metrics}
        except Exception as e:
            print(f"   ❌ {strategy} failed: {e}")

    # Print comparison table
    print(f"\n{'Strategy':<15} {'Return':>10} {'Volatility':>12} "
          f"{'Sharpe':>8} {'Drawdown':>10}")
    print("─" * 58)
    for strategy, result in results.items():
        m = result["metrics"]
        print(f"{strategy:<15} {m['expected_return']:>+10.2%} "
              f"{m['volatility']:>12.2%} {m['sharpe_ratio']:>8.2f} "
              f"{m['max_drawdown']:>10.2%}")

    return results


if __name__ == "__main__":
    from src.data_loader import get_all_data
    from src.time_series import forecast_returns
    from src.genai_features import analyze_sentiment, get_macro_signals, \
        encode_genai_features

    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    # Get all data
    data      = get_all_data(stocks)
    forecasts = forecast_returns(data["enriched"])

    # Get GenAI features
    sentiment     = analyze_sentiment(stocks, forecasts)
    macro         = get_macro_signals()
    genai_features = encode_genai_features(sentiment, macro, stocks)

    # Optimize
    weights, metrics = optimize_portfolio(
        data["returns"],
        forecasts,
        genai_features,
        strategy="max_sharpe"
    )

    # Compare strategies
    compare_strategies(data["returns"], forecasts, genai_features)