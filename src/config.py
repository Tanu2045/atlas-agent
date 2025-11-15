import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
CLEANED_DIR = DATA_DIR / "cleaned"
FAISS_DIR = DATA_DIR / "faiss_index"
TOPICS_DIR = DATA_DIR / "topics"

# Where to save final markdown reports
REPORTS_OUTPUT_DIR = ROOT_DIR / "reports_output"


# Create directories if they don't exist
for d in [DATA_DIR, RAW_HTML_DIR, CLEANED_DIR, FAISS_DIR, TOPICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Groq / LLM config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Basic safety check
if GROQ_API_KEY is None:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Create a .env file in the project root with GROQ_API_KEY=..."
    )

# General LLM defaults
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
