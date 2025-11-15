import hashlib
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
import trafilatura

from ..config import CLEANED_DIR


class ExtractionAgent:
    """
    Cleans raw HTML and extracts main article text / content.
    Stores cleaned text under data/cleaned.
    """

    def __init__(self) -> None:
        # You could add options later (e.g., min length)
        pass

    def _filename_for_url(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return CLEANED_DIR / f"{h}.txt"

    def extract(self, html: str, url: Optional[str] = None) -> str:
        """
        Extract readable text from HTML.
        Uses trafilatura first, falls back to BeautifulSoup.
        Optionally saves to disk if URL is provided.
        """
        text = self._extract_with_trafilatura(html)
        if not text:
            text = self._fallback_with_bs4(html)

        text = text.strip()

        if url is not None and text:
            out_path = self._filename_for_url(url)
            out_path.write_text(text, encoding="utf-8")

        return text

    @staticmethod
    def _extract_with_trafilatura(html: str) -> str:
        # trafilatura.extract returns None if it can't extract
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        return extracted or ""

    @staticmethod
    def _fallback_with_bs4(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n")
