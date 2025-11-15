from __future__ import annotations

from typing import List, Dict

from ..llm_client import GroqLLMClient


SYSTEM_PROMPT = """
You are the Answer Agent for ATLAS, a research assistant focused on accurate,
evidence-based answers.

You will be given:
- a question
- a set of numbered evidence chunks from web documents

Your job:
- Use ONLY the provided evidence to answer the question.
- Cite chunks inline like [1], [2] where:
  - [1] refers to evidence chunk 1
  - [2] refers to evidence chunk 2
- If the evidence is insufficient to answer the query exactly, you MUST:
  1. Clearly state the limitation,
  2. Use whatever partial information *is* available to give a best-effort, approximate,
     or indirect answer,
  3. Stay strictly grounded in the evidence,
  4. Never invent new numbers or facts.
  
- You MAY give approximate statements like:
  “About one-third…”, “a majority…”, “some companies…”,  
  **ONLY if the evidence supports the approximation.**

- Your goal is to provide the most useful, evidence-grounded answer,
  even if the evidence is incomplete.
- When a question expects a percentage but exact data is not present, summarize the
  closest available statistics and explicitly explain what they refer to.

Answer clearly and concisely in a few paragraphs.
""".strip()


class AnswerAgent:
    def __init__(self, llm_client: GroqLLMClient | None = None) -> None:
        self.llm = llm_client or GroqLLMClient()

    def answer(self, question: str, evidence_chunks: List[Dict]) -> str:
        """
        `evidence_chunks` is a list of dicts with keys: id, text, url, rank, score
        """
        if not evidence_chunks:
            return "I don't have any evidence to answer this question yet."

        evidence_block_lines = []
        for i, ch in enumerate(evidence_chunks, start=1):
            text = ch.get("text", "")
            url = ch.get("url") or "unknown source"
            # Truncate super-long chunks to keep the prompt manageable
            text_preview = text[:1200]
            evidence_block_lines.append(
                f"[{i}] (source: {url})\n{text_preview}"
            )

        evidence_block = "\n\n".join(evidence_block_lines)

        user_content = f"""
Question:
{question}

Evidence chunks:
{evidence_block}

Please answer the question, using inline citations like [1], [2] that refer
to the evidence chunks above.
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return self.llm.chat(messages)
