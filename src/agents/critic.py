from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import json

from ..llm_client import GroqLLMClient


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent for ATLAS, an autonomous research system.

Your job:
- Check whether a given answer is well supported by a set of evidence chunks.
- Be STRICT. Do NOT assume facts that are not clearly stated in the evidence.
- If the answer goes beyond what the evidence supports, mark those parts as unsupported.

You MUST respond ONLY with a single JSON object, no extra commentary, with the schema:
{
  "faithfulness_score": <number between 0 and 1>,
  "unsupported_claims": [<string>, ...],
  "verdict": "pass" | "fail",
  "rationale": "<short explanation>"
}

Guidelines:
- "faithfulness_score" near 1.0: answer is strongly grounded in evidence.
- Around 0.5: mixed; some claims supported, others not.
- Near 0.0: largely unsupported or contradictory to evidence.
- "unsupported_claims": list of specific sentences/claims from the answer that are
  not directly supported by the evidence. If none, use an empty list [].
- "verdict" should be:
  - "pass" if the answer is mostly faithful (score >= 0.7)
  - "fail" otherwise.
- "rationale": 1-4 sentences explaining your reasoning.
""".strip()


@dataclass
class CriticResult:
    faithfulness_score: float
    unsupported_claims: List[str]
    verdict: str
    rationale: str

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "CriticResult":
        # Defensive parsing with defaults
        score = raw.get("faithfulness_score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        unsupported = raw.get("unsupported_claims") or []
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]

        verdict = raw.get("verdict") or "fail"
        rationale = raw.get("rationale") or ""

        return cls(
            faithfulness_score=score,
            unsupported_claims=[str(x) for x in unsupported],
            verdict=str(verdict),
            rationale=str(rationale),
        )


class CriticAgent:
    """
    Uses the LLM to check how well an answer is supported by evidence.
    """

    def __init__(
        self,
        llm_client: GroqLLMClient | None = None,
        default_threshold: float = 0.7,
        max_evidence_chars: int = 6000,
    ) -> None:
        self.llm = llm_client or GroqLLMClient()
        self.default_threshold = default_threshold
        self.max_evidence_chars = max_evidence_chars

    def _build_evidence_block(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a compact text block of numbered evidence snippets.
        """
        parts: List[str] = []
        total = 0

        for i, ch in enumerate(evidence_chunks, start=1):
            url = ch.get("url") or "N/A"
            text = (ch.get("text") or "").strip()
            snippet = text[:800]
            entry = f"[{i}] Source: {url}\n{text}\n"
            entry_len = len(entry)

            if total + entry_len > self.max_evidence_chars:
                # Truncate and stop
                remaining = max(0, self.max_evidence_chars - total)
                entry = entry[:remaining] + "\n...[evidence truncated]..."
                parts.append(entry)
                break

            parts.append(entry)
            total += entry_len

        return "\n\n".join(parts)

    def critique(
        self,
        question: str,
        answer: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> CriticResult:
        """
        Main entry point:
        - question: the subquestion being answered
        - answer: the model's answer (with citations)
        - evidence_chunks: list of chunks used to answer
        """
        evidence_block = self._build_evidence_block(evidence_chunks)

        user_content = (
            "Question:\n"
            f"{question}\n\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Evidence chunks (numbered):\n"
            f"{evidence_block}\n\n"
            "Now return ONLY the JSON object as specified in the system prompt."
        )

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_text = self.llm.chat(messages)

        # Try to parse JSON robustly
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to salvage a JSON substring if the model misbehaved
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    parsed = {}

        if not parsed:
            # Fallback conservative result
            return CriticResult(
                faithfulness_score=0.0,
                unsupported_claims=[
                    "CriticAgent could not parse JSON response from LLM."
                ],
                verdict="fail",
                rationale="Failed to parse the critic JSON response.",
            )

        return CriticResult.from_raw(parsed)
