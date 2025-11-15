from __future__ import annotations

from typing import Any, Dict, List

from ..llm_client import GroqLLMClient


SYSTEM_PROMPT = """
You are the Report Composer for ATLAS, an autonomous research system.

You will be given:
- the original user query
- a structured research plan (tasks + subquestions)
- a set of subquestion answers with evidence snippets and source URLs

Your job is to write a clear, well-structured MARKDOWN research report.

The report MUST:
- Start with: # <short, clear title>
- Then: ## Abstract — 1 short paragraph
- Then: ## Introduction — explain the question and context
- Then: ## Findings
  - For each major theme or task, create a subsection:
    - ### <theme or task name>
    - Summarize the relevant subquestion answers.
    - Use concise paragraphs and bullet lists where helpful.
- If comparisons are natural (e.g., between regions, models, or time periods),
  add a simple markdown table under the relevant section.
- Then: ## Risks, Limitations, and Uncertainties
  - Note where evidence is limited, conflicting, or speculative.
- Then: ## Conclusion
  - 1–3 paragraphs that answer the overall query.
- Then: ## References
  - Bullet list of distinct sources (URLs or site + title) deduplicated.

Important:
- Be faithful to the provided answers and evidence.
- Do NOT invent statistics or facts that are not supported.
- It is OK to say "Research did not find strong evidence on X".
- Do NOT include raw JSON or internal data structures in the report.
- Use neutral, analytic tone.
""".strip()


class ReportComposer:
    def __init__(self, llm_client: GroqLLMClient | None = None) -> None:
        self.llm = llm_client or GroqLLMClient()

        # Hard caps to avoid 413 errors (empirically safe)
        self.max_plan_chars = 4000          # ~1–2K tokens
        self.max_answer_chars = 1200        # per subquestion
        self.max_answers_block_chars = 12000  # total answers block
        self.max_answers_used = 20          # limit number of subanswers sent

    def _truncate(self, text: str, max_chars: int, note: str = "") -> str:
        if len(text) <= max_chars:
            return text
        suffix = f"\n...[truncated {note}]..." if note else "\n...[truncated]..."
        return text[: max_chars - len(suffix)] + suffix

    def _summarize_answers_for_prompt(
        self, answers: List[Dict[str, Any]]
    ) -> str:
        """
        Turn the list of subquestion answers into a compact text block
        that is easy for the LLM to consume, with aggressive truncation.
        """
        lines: List[str] = []

        # Limit how many answers we include
        answers_to_use = answers[: self.max_answers_used]

        for i, item in enumerate(answers_to_use, start=1):
            task_id = item.get("task_id") or "?"
            task_desc = (item.get("task_description") or "").strip()
            subq = (item.get("subquestion") or "").strip()
            answer = (item.get("answer") or "").strip()
            critic = item.get("critic") or {}
            critic_score = critic.get("faithfulness_score")
            critic_verdict = critic.get("verdict")


            # Truncate long answers
            answer = self._truncate(
                answer,
                self.max_answer_chars,
                note="answer text for this subquestion",
            )

            # Collect a few distinct URLs from evidence
            evidence = item.get("evidence") or []
            urls: List[str] = []
            for ch in evidence:
                url = ch.get("url")
                if url and url not in urls:
                    urls.append(url)
                if len(urls) >= 3:  # keep up to 3 per subquestion
                    break

            lines.append(
                f"### Answer {i}\n"
                f"- Task: {task_id} — {task_desc}\n"
                f"- Subquestion: {subq}\n"
                f"- Answer (with inline citations):\n{answer}\n"
                f"- Critic faithfulness_score: {critic_score} (verdict: {critic_verdict})\n"
                f"- Evidence sources:\n"
                + "\n".join(f"  - {u}" for u in urls)
                + "\n"
            )


        full_block = "\n".join(lines)
        # Truncate the full block if still huge
        full_block = self._truncate(
            full_block,
            self.max_answers_block_chars,
            note="overall answers block",
        )
        return full_block

    def compose_report(
        self,
        user_query: str,
        plan: Dict[str, Any],
        answers: List[Dict[str, Any]],
    ) -> str:
        """
        Main entry: produce a markdown research report.
        """
        if not answers:
            return (
                "# Report\n\n"
                "No answers were generated; there may have been issues with "
                "web search, scraping, or retrieval."
            )

        plan_json = json_dumps_safe(plan)
        # Truncate plan JSON if needed
        plan_json = self._truncate(
            plan_json,
            self.max_plan_chars,
            note="research plan JSON",
        )

        answers_block = self._summarize_answers_for_prompt(answers)

        user_content = (
            f"User query:\n{user_query}\n\n"
            f"Research plan (JSON, possibly truncated):\n{plan_json}\n\n"
            "Subquestion answers and evidence (possibly truncated):\n\n"
            f"{answers_block}\n\n"
            "Now write the final markdown research report as specified."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Let llm_client handle HTTP errors in a generic way
        return self.llm.chat(messages)


def json_dumps_safe(obj: Any) -> str:
    """
    Small helper to stringify the plan in a stable way without crashing
    if it contains non-serializable values.
    """
    import json

    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except TypeError:
        # Fallback: coerce everything to string
        def _stringify(x: Any):
            if isinstance(x, (dict, list, tuple)):
                if isinstance(x, dict):
                    return {str(k): _stringify(v) for k, v in x.items()}
                return [_stringify(v) for v in x]
            return str(x)

        return json.dumps(_stringify(obj), indent=2, ensure_ascii=False)
