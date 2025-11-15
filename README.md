ATLAS: Autonomous Research Agent (MVP)

ATLAS is a small autonomous research agent that can take a natural-language query, plan a research strategy, search the web, retrieve evidence, and generate a well-structured, citation-rich markdown report.

It’s designed as a portfolio-ready project to showcase:

Agentic LLM workflows

Retrieval-Augmented Generation (RAG)

Web search + scraping + document cleaning

Evidence-grounded answering (with optional self-critique)

Clean, modular Python architecture

Features

End-to-end pipeline in one command

python -m src.main "your research question" --with-search


Research planner
Decomposes the user query into tasks and subquestions.

Web search + RAG index

Uses web search to fetch relevant pages.

Cleans HTML into plain text files under data/cleaned/.

Builds a lightweight local retrieval index over cleaned pages.

Evidence-based answering

For each subquestion, retrieves top-N chunks from the index.

Asks the LLM to answer using only that evidence.

(Optional) Runs a critic model to check for hallucinations.

Final report composer

Writes a full markdown research report with:

Title, Abstract, Introduction

Findings with subsections

Risks / Limitations

Conclusion

References (URLs)

CLI options

--with-search – enable live web search + scraping.

--max-results – limit web search result count per subquestion.

--quiet – reduce console logging.

--no-critic – disable the critic pass (faster / cheaper).

Saved outputs

Final reports saved as markdown under reports_output/.

Cleaned page texts under data/cleaned/.

Architecture

High-level components:

GroqLLMClient
Thin wrapper around the Groq /chat/completions API.

Agents (src/agents/):

planner.py – turns the user query into a structured research plan.

searcher.py – runs web search and collects raw pages.

retriever.py – builds/queries the local RAG index over cleaned pages.

answerer.py – answers each subquestion using top-K evidence chunks.

critic.py (optional) – checks answers for faithfulness to evidence.

reporter.py / report_composer.py – composes the final markdown report.

Entry point

src/main.py – CLI, orchestration of the full pipeline.

Requirements

Python 3.10+ (3.11+ recommended)

A Groq API key (free tier is enough for small tests)

Git (for version control)

Setup

Clone the repo

git clone https://github.com/<your-username>/atlas-agent.git
cd atlas-agent


Create and activate a virtual environment

python -m venv .venv


On Windows (PowerShell):

.venv\Scripts\Activate.ps1


On macOS/Linux:

source .venv/bin/activate


Install dependencies

pip install -r requirements.txt


Environment variables

Create a file named .env in the project root:

GROQ_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


(Make sure .env is in .gitignore so it never gets committed.)

Optionally you can also set:

GROQ_MODEL=llama3-70b-8192

Usage

Basic command:

python -m src.main "impact of generative AI on entry-level software engineering roles in India"


With web search enabled:

python -m src.main "impact of generative AI on entry-level software engineering roles in India" --with-search


Limit web results per subquestion (faster / cheaper):

python -m src.main "impact of generative AI on entry-level software engineering roles in India" --with-search --max-results 2


Run quietly (minimal logs):

python -m src.main "impact of generative AI on entry-level software engineering roles in India" --with-search --max-results 2 --quiet


Disable the critic (no faithfulness check, faster):

python -m src.main "impact of generative AI on entry-level software engineering roles in India" --with-search --max-results 2 --no-critic


You can also pass a multi-word query without quotes:

python -m src.main impact of generative AI on entry-level software engineering roles in India --with-search


(But using quotes is safer.)

Outputs

After a successful run you’ll get:

A markdown report under:

reports_output/
  <slug-of-query>-<timestamp>.md


This is ready to read, share, or convert to PDF.

Cleaned text data under:

data/cleaned/
  <hash1>.txt
  <hash2>.txt
  ...


Each file corresponds to a cleaned web page used as evidence.

Running Without Web Search

If you already have cleaned text files in data/cleaned/, you can run ATLAS in offline RAG-only mode:

python -m src.main "your research question"


In this mode, ATLAS:

Skips live web search.

Builds a retrieval index from whatever is in data/cleaned/.

Answers based only on that local corpus.

Project Structure (simplified)
atlas-agent/
├─ data/
│  └─ cleaned/              # cleaned text chunks from web pages
├─ reports_output/          # final markdown reports
├─ src/
│  ├─ agents/
│  │  ├─ planner.py
│  │  ├─ searcher.py
│  │  ├─ retriever.py
│  │  ├─ answerer.py
│  │  ├─ critic.py
│  │  └─ report_composer.py
│  ├─ llm_client.py         # Groq client wrapper
│  └─ main.py               # CLI entry point, pipeline orchestration
├─ .gitignore
├─ requirements.txt
└─ README.md


Design Notes

LLM client is centralized

All model calls go through GroqLLMClient, making it easy to:

Swap models.

Change temperature / max tokens.

Add logging / telemetry.

Critic is optional

By default, you can choose at CLI time whether to:

Run a second LLM pass that critiques each answer against the evidence.

Or skip it to save latency and tokens.

Aggressive truncation

The report composer and answerer truncate long context to avoid 413 / context-length issues.

For large research plans, only top-N subanswers are included in the final prompt to the reporter.
