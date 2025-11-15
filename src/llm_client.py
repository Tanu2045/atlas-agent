from typing import List, Dict
import time
import requests

from .config import GROQ_API_KEY, GROQ_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class GroqLLMClient:
    """
    Simple wrapper around Groq's OpenAI-compatible chat completions API.
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_MODEL,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> str:
        """
        Send a chat completion request and return the assistant's text.

        Retries a few times on HTTP 429 (Too Many Requests) with exponential backoff.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        last_resp = None

        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            last_resp = resp

            # If we're being rate-limited, back off and try again
            if resp.status_code == 429:
                # On final attempt, just raise
                if attempt == max_retries - 1:
                    resp.raise_for_status()

                # Use Retry-After header if present, otherwise exponential backoff
                retry_after_header = resp.headers.get("Retry-After")
                if retry_after_header is not None:
                    try:
                        delay = float(retry_after_header)
                    except ValueError:
                        delay = retry_backoff * (attempt + 1)
                else:
                    delay = retry_backoff * (attempt + 1)

                # Sleep before retrying
                time.sleep(delay)
                continue

            # For non-429 errors, raise immediately if any
            resp.raise_for_status()

            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Groq response format: {data}") from e

        # If we somehow exit the loop with no return, raise the last response error
        if last_resp is not None:
            last_resp.raise_for_status()

        raise RuntimeError("GroqLLMClient.chat failed without a valid response.")
