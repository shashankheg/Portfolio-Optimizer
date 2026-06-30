import yfinance as yf
import pandas as pd 
import numpy as np

from datetime import datetime , timedelta 
from typing import Dict, List

STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA","NOW", "NVDA", "META", "NFLX"]

def fetch_stock_data(stocks: List[str] =STOCKS,period_years :int =3)->Dict[str, pd.DataFrame] :
    """Fetch historical OHLCV data for a list of stocks.
       Args:
        stocks: List of stock tickers
        period_years: Number of years of historical data
       Returns:
        Dictionary mapping stock tickers to their historical data.
    """
    end_date =datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)
    print(f"📥 Fetching data for {len(stocks)} stocks...")
    print(f"   Period: {start_date.date()} → {end_date.date()}")
    stock_data = {}
    failed = []

    for stock in stocks:
        try:
            df = yf.download(
                stock, 
                start=start_date, 
                end=end_date,
                auto_adjust=True,
                progress=False
                )
            if df.empty:
                print(f"   ⚠️  No data for {stock} — skipping")
                failed.append(stock)
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            stock_data[stock] = df
            print(f"   ✅ {stock}: {len(df)} trading days")
            
        except Exception as e:
            print(f" ❌ {stock} failed: {e}")
            failed.append(stock)

    if failed:
        print(f"\n   ⚠️  Failed stocks: {failed}")

    print(f"\n✅ Fetched data for {len(stock_data)}/{len(stocks)} stocks")
    

    return stock_data


def compute_returns(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute daily returns for all stocks.

    Returns:
        DataFrame of daily returns — shape (days, n_stocks)
    """
    returns = pd.DataFrame({
        stock: df["Close"].pct_change().dropna()
        for stock, df in stock_data.items()
    })
    returns = returns.dropna()
    print(f"✅ Returns matrix: {returns.shape[0]} days x {returns.shape[1]} stocks")
    return returns



def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a stock DataFrame.

    Indicators: MA7, MA21, MA50, RSI, Volatility, Bollinger Bands
    """
    df = df.copy()

    # Calculate moving averages
    df['MA7']   = df['Close'].rolling(window=7).mean()
    df['MA21']  =df['Close'].rolling(window=21).mean()
    df['MA50']  =df['Close'].rolling(window=50).mean()  

    # Volatility (21-day rolling std of returns)

    df["Volatility"] = df["Close"].pct_change().rolling(21).std()

    #RSI 
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))


    #bollinger bands
    df["BB_Mid"]   = df["Close"].rolling(20).mean()

    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()

    df['BB_Lower'] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]


    #MACD ( Typically called Moving average convergence divergence calculated as 12 day and 26 day terms )

    ema12      = df["Close"].ewm(span=12).mean()
    ema26      = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    return df.dropna()



def get_all_data(stocks: List[str] = STOCKS) -> Dict:
    """
    Main function — fetch and process all data.

    Returns:
        Dict with raw data, returns, and technical indicators
    """

    #Fetch raw data
    stock_data = fetch_stock_data(stocks)

    # Compute returns
    returns = compute_returns(stock_data)



     # Add technical indicators

    enriched_data = {
        stock :compute_technical_indicators(df) for stock, df in stock_data.items()

     }
    
    return {
        "raw":        stock_data,
        "returns":    returns,
        "enriched":   enriched_data,
        "stocks":     list(stock_data.keys())
    }



if __name__ == "__main__":
    data = get_all_data()
    print("\n📊 Sample Returns:")
    print(data["returns"].tail())
    print("\n📈 Sample Technical Indicators (AAPL):")
    print(data["enriched"]["AAPL"][["Close","MA7","RSI","MACD"]].tail())



    

