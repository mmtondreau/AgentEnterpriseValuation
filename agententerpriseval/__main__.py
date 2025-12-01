import os
import dotenv
import asyncio
import agententerpriseval.stocks as stocks

dotenv.load_dotenv()

if "GOOGLE_API_KEY" in os.environ:
    print("✅ Gemini API key setup complete.")

asyncio.run(stocks.run())
