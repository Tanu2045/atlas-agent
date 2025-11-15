import hashlib
from pathlib import Path
from typing import Optional

import requests

from ..config import RAW_HTML_DIR


class ScraperAgent:
    """
    Downloads raw HTML for a given URL and stores it under data/raw_html.
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: int = 20) -> None:
        self.timeout = timeout
        # Use a modern desktop Chrome-like user agent
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        )

    def _filename_for_url(self, url: str) -> Path:
        # Simple stable hash for filename
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return RAW_HTML_DIR / f"{h}.html"

    def fetch(self, url: str) -> str:
        """
        Download URL and return raw HTML as text.
        Also saves the HTML to data/raw_html/<hash>.html
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        html = resp.text

        out_path = self._filename_for_url(url)
        out_path.write_text(html, encoding="utf-8")

        return html
