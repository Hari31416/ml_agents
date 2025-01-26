from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", None)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
LOG_LEVEL = os.getenv("LOG_LEVEL", "warning").upper()
