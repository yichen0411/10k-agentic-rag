import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

CHAT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "financials.db"
PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_DIR = BASE_DIR / "data" / "index"
