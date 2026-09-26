import os

from dotenv import load_dotenv


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# API keys
# ------------------------------------------------------------

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# ------------------------------------------------------------
# LLM provider
# ------------------------------------------------------------

# True  -> OpenAI
# False -> Cohere

USE_OPENAI = True

# Keep this name because your existing app_db.py
# currently uses "use_openai".
use_openai = USE_OPENAI