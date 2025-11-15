import argparse
import json
from typing import Any, Dict, List
import re
from datetime import datetime

from rich import print as rprint

from .agents.planner import PlannerAgent
from .agents.search import SearchAgent
from .agents.scraper import ScraperAgent
from .agents.extractor import ExtractionAgent
from .agents.answerer import AnswerAgent
from .rag.indexer import SimpleRAGIndexer
from .reports.composer import ReportComposer
from .agents.critic import CriticAgent
from .config import REPORTS_OUTPUT_DIR

def slugify(text: str, max_len: int = 60) -> str:
    """
    Turn an arbitrary query into a filesystem-friendly slug.
    """
    text = text.lower().strip()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "report"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ATLAS – Open-World Autonomous Research Agent"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="+",
        help="Research query for ATLAS to plan and research.",
    )
    parser.add_argument(
        "--with-search",
        action="store_true",
        help="After planning, run web search + RAG for all subquestions.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max search results per subquestion.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal console output.",
    )
    # ⬇️ ADD THIS
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable CriticAgent (faster runs, fewer LLM calls).",
    )

    return parser.parse_args()


def pretty_print_plan(plan: Dict[str, Any], quiet: bool) -> None:
    if quiet:
        return
    rprint("[bold cyan]=== ATLAS Research Plan ===[/bold cyan]")
    rprint(json.dumps(plan, indent=2, ensure_ascii=False))


def run_full_pipeline(
    user_query: str,
    max_results: int = 5,
    quiet: bool = False,
    use_critic: bool = True,
) -> str:

    """
    End-to-end pipeline:
    - Planner -> tasks + subquestions
    - For each subquestion: search, scrape, extract, index, retrieve, answer, critique
    - Compose final markdown report
    Returns the markdown report string.
    """
    planner = PlannerAgent()
    search_agent = SearchAgent(max_results=max_results)
    scraper = ScraperAgent()
    extractor = ExtractionAgent()
    indexer = SimpleRAGIndexer()
    answer_agent = AnswerAgent()
    critic = CriticAgent() if use_critic else None
    composer = ReportComposer()

    if not quiet:
        rprint("[bold cyan]Planning research...[/bold cyan]")

    plan = planner.plan(user_query)
    pretty_print_plan(plan, quiet)

    tasks = plan.get("tasks") or []
    if not tasks:
        if not quiet:
            rprint("[red]Planner returned no tasks; aborting.[/red]")
        return "# Report\n\nPlanner did not produce any tasks."

    all_answers: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task.get("id") or "?"
        task_desc = task.get("description") or ""
        subquestions = task.get("subquestions") or []

        if not quiet:
            rprint(f"\n[bold green]=== Task {task_id}: {task_desc} ===[/bold green]")

        for subq in subquestions:
            if not quiet:
                rprint(f"\n[bold]Subquestion:[/bold] {subq}")
                rprint("[bold cyan]Searching web...[/bold cyan]")

            # 1) Search
            results = search_agent.search(subq)
            if not results:
                if not quiet:
                    rprint("[yellow]No search results found.[/yellow]")
                continue

            # 2) Scrape + extract + index
            cleaned_docs: List[Dict[str, Any]] = []
            for res in results:
                url = res.get("href") or res.get("url")
                title = res.get("title") or ""
                snippet = res.get("snippet") or ""

                if not url:
                    continue

                if not quiet:
                    rprint(f"- [link={url}]{title or url}[/link]")
                    if snippet:
                        rprint(f"  {snippet}")

                try:
                    html = scraper.fetch(url)
                except Exception as e:
                    if not quiet:
                        rprint(f"[yellow]Error fetching URL, skipping:[/yellow] {e}")
                    continue

                text = extractor.extract(html, url=url)
                if not text.strip():
                    if not quiet:
                        rprint("[yellow]No meaningful text extracted, skipping.[/yellow]")
                    continue

                cleaned_docs.append({"url": url, "title": title, "text": text})

                # Add to RAG index
                indexer.index_document(text, url=url)

            if not cleaned_docs:
                if not quiet:
                    rprint("[yellow]No cleaned documents; skipping.[/yellow]")
                continue

            # 3) Retrieve evidence
            if not quiet:
                rprint("[bold cyan]Retrieving evidence from RAG index...[/bold cyan]")

            top_chunks = indexer.retrieve(subq, k=5)
            if not top_chunks:
                if not quiet:
                    rprint("[yellow]No chunks retrieved; cannot answer.[/yellow]")
                continue

            if not quiet:
                rprint("[bold green]Top evidence chunks:[/bold green]")
                for ch in top_chunks:
                    preview = ch["text"][:300].replace("\n", " ")
                    rprint(
                        f"[bold][{ch['rank']}][/bold] score={ch['score']:.3f} (source: {ch.get('url')})\n{preview}...\n"
                    )

            # 4) Answer
            if not quiet:
                rprint("[bold cyan]Answering subquestion using evidence...[/bold cyan]")

            answer_text = answer_agent.answer(subq, top_chunks)

            if not quiet:
                rprint("[bold magenta]Subquestion answer:[/bold magenta]")
                rprint(answer_text)

            # 5) Critic
            critic_result = None
            if critic is not None:
                if not quiet:
                    rprint("[bold cyan]Critiquing answer for faithfulness...[/bold cyan]")

                critic_result = critic.critique(subq, answer_text, top_chunks)

                if not quiet:
                    rprint(
                        f"[bold yellow]Critic faithfulness score:[/bold yellow] "
                        f"{critic_result.faithfulness_score:.2f} "
                        f"(verdict: {critic_result.verdict})"
                    )
                    if critic_result.unsupported_claims:
                        rprint("[yellow]Potential unsupported claims:[/yellow]")
                        for c in critic_result.unsupported_claims:
                            rprint(f" - {c}")
                    if critic_result.rationale:
                        rprint(f"[dim]Critic rationale: {critic_result.rationale}[/dim]")

            # ⬇️ OUTSIDE the if critic is not None: block
            all_answers.append(
                {
                    "task_id": task_id,
                    "task_description": task_desc,
                    "subquestion": subq,
                    "answer": answer_text,
                    "evidence": top_chunks,
                    "critic": {
                        "faithfulness_score": getattr(critic_result, "faithfulness_score", None),
                        "unsupported_claims": getattr(critic_result, "unsupported_claims", []),
                        "verdict": getattr(critic_result, "verdict", None),
                        "rationale": getattr(critic_result, "rationale", ""),
                    } if critic_result is not None else None,
                }
            )



    # 6) Compose final report
    if not quiet:
        rprint("\n[bold cyan]Composing final markdown report...[/bold cyan]")

    report_md = composer.compose_report(user_query, plan, all_answers)
    return report_md


def main() -> None:
    args = parse_args()
    user_query = " ".join(args.query)

    if not args.with_search:
        planner = PlannerAgent()
        plan = planner.plan(user_query)
        pretty_print_plan(plan, args.quiet)
        if not args.quiet:
            rprint(
                "\n[dim](Pass --with-search to run web search, RAG, and report generation.)[/dim]"
            )
        return

    report_md = run_full_pipeline(
        user_query,
        max_results=args.max_results,
        quiet=args.quiet,
        use_critic=not args.no_critic,
    )

    # Ensure output directory exists
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a filename from the query + timestamp
    slug = slugify(user_query)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{slug}-{timestamp}.md"
    output_path = REPORTS_OUTPUT_DIR / filename

    # Save report to disk
    output_path.write_text(report_md, encoding="utf-8")

    if not args.quiet:
        rprint("\n[bold magenta]=== FINAL ATLAS REPORT (Markdown) ===[/bold magenta]\n")
        rprint(report_md)
        rprint(f"\n[green]Saved report to:[/green] {output_path}")
    else:
        # In quiet mode, just tell you where it went
        print(f"Saved report to: {output_path}")



if __name__ == "__main__":
    main()
