import json
from typing import Any, Dict

from ..llm_client import GroqLLMClient


SYSTEM_PROMPT = """
You are the Planner Agent for ATLAS, an autonomous research system.

Your job:
- Break a user research query into a small set of research tasks.
- For each task, define:
  - id (short string)
  - description
  - subquestions (list of specific questions)
  - document_types (e.g. "blog posts", "academic papers", "news", "docs")

Return ONLY valid JSON in this schema:

{
  "overall_goal": "<copy or refine the user query>",
  "tasks": [
    {
      "id": "t1",
      "description": "...",
      "subquestions": ["...", "..."],
      "document_types": ["...", "..."]
    }
  ]
}
""".strip()


class PlannerAgent:
    def __init__(self, llm_client: GroqLLMClient | None = None) -> None:
        self.llm = llm_client or GroqLLMClient()

    def plan(self, user_query: str) -> Dict[str, Any]:
        """
        Produce a structured research plan as a Python dict.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Create a research plan for this query:\n{user_query}",
            },
        ]

        raw = self.llm.chat(messages)

        # Try to parse JSON strictly, but fall back to a crude recovery if needed.
        plan = self._parse_json_safely(raw)
        return plan

    @staticmethod
    def _parse_json_safely(text: str) -> Dict[str, Any]:
        """
        Try strict JSON parse; if it fails, try to recover the first JSON object.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # naive recovery: find first '{' and last '}' and parse that slice
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass

        # If everything fails, return a very simple fallback plan
        return {
            "overall_goal": text[:200],
            "tasks": [
                {
                    "id": "t1",
                    "description": "Fallback single task based on the query.",
                    "subquestions": [],
                    "document_types": [],
                }
            ],
        }
