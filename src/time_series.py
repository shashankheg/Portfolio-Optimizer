import numpy as np
import pandas as pd
from typing import Dict ,List
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_loader import get_all_data



def compute_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose time series into trend, seasonality, and residual components.

    Args:
        df: Stock DataFrame with technical indicators

    Returns:
        DataFrame with time series features added
    """
    df = df.copy()
    close = df["Close"]

    # === TREND ===
    # Linear trend over 30-day window
    df["Trend"] = close.rolling(30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]
        if len(x) == 30 else np.nan
    )


    # Trend direction
    df["Trend_Direction"] = df["Trend"].apply(lambda x: 1 if x > 0 else -1)

    # === SEASONALITY ===
    # Day of week effect (0=Monday, 4=Friday)

    df["DayOfWeek"]  = df.index.dayofweek
    df ['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

# Weekly seasonality — average return by day of week

    returns = close.pct_change()
    df["Returns"] = returns
    dow_avg = returns.groupby(df.index.dayofweek).transform("mean")
    df["DayOfWeek_Seasonality"] = dow_avg


 # Monthly seasonality
    month_avg = returns.groupby(df.index.month).transform("mean")
    df["Month_Seasonality"] = month_avg

    # === MOMENTUM ===
    df["Momentum_5"]  = close.pct_change(5)    # 5-day momentum
    df["Momentum_21"] = close.pct_change(21)   # 21-day momentum
    df["Momentum_63"] = close.pct_change(63)   # 63-day momentum (quarter)


 # === MEAN REVERSION ===
    df["Z_Score_21"] = (
        (close - close.rolling(21).mean()) /
        close.rolling(21).std()
    )

    for lag in [1,2,3,5,10]:
        df[f"Lag_{lag}"] = returns.shift(lag)
    
    return df.dropna()

def forecast_stock(
        df: pd.DataFrame,
        stock : str,
        horizon_days: int =30
    ) -> Dict:

    """
    Forecast stock return and volatility using ML model.

    Args:
        df: Enriched stock DataFrame with all features
        stock: Stock ticker name
        horizon_days: Forecast horizon in days

    Returns:
        Dict with return forecast, volatility, confidence, direction
    """
    df = compute_time_series_features(df)

    # Feature columns
    feature_cols = [
        "MA7", "MA21", "MA50", "RSI", "Volatility",
        "MACD", "BB_Width", "Trend", "Trend_Direction",
        "DayOfWeek_Seasonality", "Month_Seasonality",
        "Momentum_5", "Momentum_21", "Z_Score_21",
        "Lag_1", "Lag_2", "Lag_3", "Lag_5"
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Target: forward return over horizon
    df["Target"] = df["Close"].pct_change(horizon_days).shift(-horizon_days)
    df = df.dropna()


    if len(df) < 100:
        return _fallback_forecast(stock)
    
    X = df[feature_cols].values
    y = df["Target"].values

    # Train test split 
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    scaler  = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Gradient Boosting model

    model = GradientBoostingRegressor(
        n_estimators=100, 
        random_state=42,
        
        learning_rate=0.05,
        max_depth=3
)

    model.fit(X_train, y_train)
    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    print(f"Test MSE: {mse}")
    print(f"Test MAE: {mae}")

# Predict next period using latest data
    latest_features = scaler.transform(X[-1:])
    predicted_return = model.predict(latest_features)[0]
    

# Confidence based on model accuracy
    actual_std  = np.std(y_test)
    confidence  = max(0.3, min(0.95, 1 - (rmse / (actual_std + 1e-8))))

    # Volatility forecast
    recent_returns  = df["Returns"].tail(21)
    volatility      = float(recent_returns.std() * np.sqrt(252))
# Direction
    direction = "up" if predicted_return > 0.01 else \
                "down" if predicted_return < -0.01 else "sideways"
    
    print(f"   📈 {stock}: return={predicted_return:+.2%} | "
          f"vol={volatility:.2%} | conf={confidence:.2f} | "
          f"RMSE={rmse:.4f}")
    return {
        "return":     predicted_return,
        "volatility": volatility,
        "confidence": confidence,
        "direction":  direction,
        "rmse":       rmse,
        "mae":        mae,
        "horizon_days": horizon_days
    }


def forecast_returns(
    stock_data: Dict[str, pd.DataFrame],
    horizon_days: int = 30
) -> Dict[str, Dict]:
    """
    Forecast returns for all stocks.

    Args:
        stock_data: Dict of enriched stock DataFrames
        horizon_days: Forecast horizon

    Returns:
        Dict of {ticker: forecast_dict}
    """
    print(f"\n📊 Forecasting {len(stock_data)} stocks "
          f"({horizon_days}-day horizon)...")

    forecasts = {}
    for stock, df in stock_data.items():
        try:
            forecasts[stock] = forecast_stock(df, stock, horizon_days)
        except Exception as e:
            print(f"   ❌ {stock} forecast failed: {e}")
            forecasts[stock] = _fallback_forecast(stock)

    return forecasts


def _fallback_forecast(stock: str) -> Dict:
    """Return a neutral fallback forecast when model fails."""
    print(f"   ⚠️  {stock}: Using fallback forecast")
    return {
        "return":       0.0,
        "volatility":   0.20,
        "confidence":   0.30,
        "direction":    "sideways",
        "rmse":         None,
        "mae":          None,
        "horizon_days": 30
    }

if __name__ == "__main__":
    from src.data_loader import get_all_data

    data      = get_all_data()
    forecasts = forecast_returns(data["enriched"])

    print("\n📋 Forecast Summary:")
    print(f"{'Stock':<8} {'Return':>10} {'Volatility':>12} "
          f"{'Confidence':>12} {'Direction':>10}")
    print("─" * 56)
    for stock, f in forecasts.items():
        print(f"{stock:<8} {f['return']:>+10.2%} {f['volatility']:>12.2%} "
              f"{f['confidence']:>12.2f} {f['direction']:>10}")

    





