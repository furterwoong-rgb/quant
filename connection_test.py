"""
Alpaca Paper Trading connection test.
Run this first to verify API keys and connectivity.
"""
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

def test_connection():
    print("=" * 50)
    print("Alpaca Paper Trading - Connection Test")
    print("=" * 50)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[FAIL] API keys not found. Check config/.env file.")
        return False

    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = trading_client.get_account()

        print(f"[OK] Connected to Alpaca Paper Trading")
        print(f"     Account status : {account.status}")
        print(f"     Portfolio value : ${float(account.portfolio_value):,.2f}")
        print(f"     Cash            : ${float(account.cash):,.2f}")
        print(f"     Buying power    : ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"[FAIL] Trading client error: {e}")
        return False

    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=5),
            end=datetime.now() - timedelta(days=1),
        )
        bars = data_client.get_stock_bars(request)
        df = bars.df
        print(f"\n[OK] Market data feed working")
        print(f"     SPY last close  : ${df['close'].iloc[-1]:.2f}")
        print(f"     Data points     : {len(df)} bars")
    except Exception as e:
        print(f"[FAIL] Data client error: {e}")
        return False

    print("\n[READY] All systems operational. Ready to trade.")
    return True

if __name__ == "__main__":
    test_connection()
