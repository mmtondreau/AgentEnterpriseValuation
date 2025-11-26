from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

eodHistoricalData = McpToolset(
    connection_params=StreamableHTTPServerParams(
        url="http://127.0.0.1:8000/mcp",
        timeout=60,
    ),
    tool_filter=[
        # Core EODHD datasets
        "get_historical_stock_prices",
        "get_live_price_data",
        "get_intraday_historical_data",
        "get_fundamentals_data",
        "get_us_tick_data",
        "get_historical_market_cap",
        "get_company_news",
        "get_sentiment_data",
        "get_news_word_weights",
        "get_exchanges_list",
        "get_exchange_tickers",
        "get_exchange_details",
        "get_macro_indicator",
        "get_economic_events",
        "get_symbol_change_history",
        "get_stocks_from_search",
        "get_user_details",
        "get_insider_transactions",
        "get_capture_realtime_ws",
        "get_stock_screener_data",
        "get_upcoming_earnings",
        "get_earnings_trends",
        "get_upcoming_ipos",
        "get_upcoming_splits",
        "get_dividends_data",
        "get_us_live_extended_quotes",
        "get_technical_indicators",
        # Marketplace products – EODHD
        "get_mp_us_options_contracts",
        "get_mp_us_options_eod",
        "get_mp_us_options_underlyings",
        "get_mp_indices_list",
        "get_mp_index_components",
        # Marketplace products – third party providers
        "get_mp_illio_performance_insights",
        "get_mp_illio_risk_insights",
        "get_mp_illio_market_insights_performance",
        "get_mp_illio_market_insights_best_worst",
        "get_mp_illio_market_insights_volatility",
        "get_dividends_data",
    ],
)
