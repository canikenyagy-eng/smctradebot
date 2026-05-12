# MetaTrader 5 Data Provider Integration

This document describes how to integrate MetaTrader 5 as an optional market data source into the existing trading system.

## Overview

The system now supports two market data providers:

1. **Yahoo Finance** (default, existing) - Free data from Yahoo
2. **MetaTrader 5** (new) - Professional data from MT5 terminal

## Installation

### Install MetaTrader5 Python Package

```bash
pip install MetaTrader5
```

### Requirements

- MetaTrader 5 terminal installed on Windows
- Valid MT5 account with broker
- Python 3.8+

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Data source selection (yahoo or mt5)
DATA_SOURCE=mt5

# MT5 connection settings
MT5_LOGIN=123456
MT5_PASSWORD=your_password
MT5_SERVER=YourBrokerServer
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
```

### Configuration Details

| Variable | Description | Default |
|----------|-------------|---------|
| DATA_SOURCE | Data provider ("yahoo" or "mt5") | "yahoo" |
| MT5_LOGIN | MT5 account login number | (none) |
| MT5_PASSWORD | MT5 account password | (none) |
| MT5_SERVER | MT5 broker server | (none) |
| MT5_PATH | Path to MT5 terminal exe | "C:/Program Files/MetaTrader 5/terminal64.exe" |

## Usage

### Using MT5 Data Provider

1. Set `DATA_SOURCE=mt5` in your `.env` file
2. Configure MT5 credentials
3. Run the system normally

The system will automatically use MT5 for market data when configured.

### Timeframe Mapping

The following timeframes are supported:

| SMC Timeframe | MT5 Constant | Description |
|---------------|--------------|-------------|
| M1 | 1 | 1 minute |
| M5 | 5 | 5 minutes |
| M15 | 15 | 15 minutes |
| M30 | 30 | 30 minutes |
| H1 | 16392 | 1 hour |
| H4 | 16396 | 4 hours |
| D1 | 16408 | 1 day |

## Live Mode (Optional)

For real-time data polling with MT5:

```python
from data.mt5_provider import create_mt5_provider, create_live_polling_provider

# Create provider
provider = create_mt5_provider(
    login=123456,
    password="your_password",
    server="YourBrokerServer",
)

# Create live polling
live = create_live_polling_provider(
    provider=provider,
    symbols=["EURUSD", "GBPUSD"],
    timeframe="M1",
    interval_seconds=60,
    callback=lambda symbol, df: print(f"New data for {symbol}"),
)

# Start polling
live.start()

# Get latest data
df = live.get_latest("EURUSD")

# Stop polling
live.stop()
```

## Error Handling

The system handles the following error scenarios:

1. **MT5 not installed**: Falls back to Yahoo
2. **Terminal not running**: Falls back to Yahoo
3. **Authentication failure**: Logs error, falls back to Yahoo
4. **Connection drop**: Retries connection, falls back to Yahoo

## Backtest Compatibility

The MT5 data provider can be used for backtesting by setting `DATA_SOURCE=mt5`. Historical data from MT5 is used the same way as Yahoo data.

## Programmatic Usage

### Direct Provider Usage

```python
from data.provider_factory import get_default_manager

# Create manager with MT5
manager = get_default_manager(
    data_source="mt5",
    mt5_config={
        "login": 123456,
        "password": "password",
        "server": "server",
    },
    history_limit=500,
)

# Fetch OHLCV data
df = manager.fetch_ohlcv("EURUSD", "M5", limit=100)
print(df.head())
```

### Using MarketDataClient

```python
from data.market_data import MarketDataClient

# Create client with MT5
client = MarketDataClient(
    history_limit=500,
    data_source="mt5",
    mt5_login=123456,
    mt5_password="password",
    mt5_server="server",
)

# Fetch data (same API as before)
df = client.fetch_ohlcv("EURUSD", "M5", limit=100)
```

## Architecture

```
data/
├── market_data_base.py     # Abstract interface
├── mt5_provider.py     # MT5 implementation
├── yahoo_provider.py  # Yahoo implementation
├── provider_factory.py # Provider management
└── market_data.py     # Backward-compatible client
```

## Security Notes

-Store MT5 credentials securely in `.env` file
- Never commit credentials to version control
- Use environment variables for production deployments

## Troubleshooting

### MT5 Connection Fails

1. Verify terminal is running
2. Check login/password/server are correct
3. Ensure MetaTrader5 package is installed: `pip install MetaTrader5`

### No Data Returns

1. Check symbol is available in MT5
2. Verify timeframe is supported
3. Check MT5 terminal logs for errors

### Falls Back to Yahoo

This is expected behavior when MT5 fails. The system will log a warning and automatically use Yahoo data.