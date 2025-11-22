# Financial Assistant Agent

A Google ADK agent that provides financial analysis using real-time market data from the EODHD (EOD Historical Data) service.

## Features

- **Real-time Market Data**: Access to stock prices, fundamentals, news, and more via EODHD MCP server
- **Long-term Memory**: Remembers past analyses and can reference them in future conversations
- **Comprehensive Analysis**: Provides detailed financial analysis including:
  - Company fundamentals (P/E ratios, market cap, revenue, etc.)
  - Recent news and sentiment
  - Analyst ratings and price targets
  - Historical performance metrics

## Tools Available

The agent has access to the following tools from the EODHD MCP server:

- `get_historical_stock_prices` - Historical price data
- `get_fundamentals_data` - Company fundamental data
- `get_company_news` - Recent news articles
- `get_sentiment_data` - Market sentiment analysis
- `get_macro_indicator` - Macroeconomic indicators
- `get_economic_events` - Upcoming economic events
- `get_upcoming_earnings` - Earnings calendar

## Configuration

The agent requires:
1. EODHD MCP server running on `http://127.0.0.1:8000/mcp`
2. PostgreSQL database for session and memory storage
3. Google Gemini API key for the LLM

## Usage Examples

Try asking:
- "Get the latest fundamental data on AAPL and provide a summary of its recent performance."
- "What's the current sentiment on Tesla stock?"
- "Show me upcoming earnings for tech companies this week."
- "Compare the fundamentals of MSFT and GOOGL."

## Model

Uses `gemini-2.5-flash-lite` for cost-efficient and fast responses.
