from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from ddgs import DDGS


class SearchAgent:
    """
    Uses DuckDuckGo (via the `ddgs` package) to search the web for a given query.

    We aggressively filter out:
    - obvious ads / tracking URLs
    - known junk domains (baidu zhidao, biggerpockets, etc.)
    - results that don't even mention AI / generative / software / India

    This makes results fewer, but more relevant for our use case.
    """

    def __init__(
        self,
        max_results: int = 20,
        region: str = "in-en",  # India + English bias
    ) -> None:
        self.max_results = max_results
        self.region = region

        # Keywords we care about (for filtering)
        self.required_keywords = [
            "ai",
            "artificial intelligence",
            "generative",
            "llm",
            "large language model",
            "software",
            "developer",
            "engineering",
            "india",
            "indian",
        ]

        # Domains we know are junk for our use case
        self.blacklist_domains = [
            "duckduckgo.com",        # ads / tracking
            "zhidao.baidu.com",      # Chinese Q&A pages
            "biggerpockets.com",     # real estate
        ]

    def _is_blacklisted(self, url: Optional[str]) -> bool:
        if not url:
            return True
        host = urlparse(url).netloc.lower()
        return any(bad in host for bad in self.blacklist_domains)

    def _has_relevant_keywords(self, title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        return any(kw in text for kw in self.required_keywords)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Run a rewritten query and filter out irrelevant / junk results.
        """
        # Force core terms into the query to steer DuckDuckGo
        rewritten_query = f'{query} "generative AI" "India" software companies'

        results: List[Dict[str, Any]] = []

        with DDGS() as ddgs:
            for r in ddgs.text(
                rewritten_query,
                region=self.region,
                max_results=self.max_results,
            ):
                url = r.get("href") or r.get("url")
                title = r.get("title") or ""
                snippet = r.get("body") or r.get("description") or ""

                # Drop ad / tracking / junk domains
                if self._is_blacklisted(url):
                    continue

                # Drop results that clearly aren't about AI / software / India at all
                if not self._has_relevant_keywords(title, snippet):
                    continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "duckduckgo",
                    }
                )

                # We only really need the top few *good* ones
                if len(results) >= 5:
                    break

        return results
