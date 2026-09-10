"""
Loads configuration (API keys, model names) from .env, with sensible
defaults where possible.

Single place all other modules read configuration from, so changing an env
var name or default only requires editing this file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root, regardless of the current working directory.
load_dotenv(Path(__file__).parent.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
