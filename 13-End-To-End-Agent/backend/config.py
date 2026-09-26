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


# ------------------------------------------------------------
# LLM provider
# ------------------------------------------------------------

# True  -> OpenAI
# False -> Cohere

USE_OPENAI = False

# Keep this name because your existing app_db.py
# currently uses "use_openai".
use_openai = USE_OPENAI