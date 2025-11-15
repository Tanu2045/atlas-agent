# ATLAS: Autonomous Research Agent (MVP)

ATLAS is a small autonomous research agent that can take a natural-language query, plan a research strategy, search the web, retrieve evidence, and generate a well-structured, citation-rich markdown report.

It is designed as a portfolio-ready project to showcase:

- Agentic LLM workflows
- Retrieval-Augmented Generation (RAG)
- Web search + scraping + document cleaning
- Evidence-grounded answering (with optional self-critique)
- Clean, modular Python architecture


## Features

### End-to-end pipeline in one command

```bash
python -m src.main "your research question" --with-search
```

## Research planner

Decomposes the user query into tasks and subquestions.


## Web search + RAG index

- Uses web search to fetch relevant pages.
- Cleans HTML into plain text files under `data/cleaned/`.
- Builds a lightweight local retrieval index over cleaned pages.


## Evidence-based answering

- Retrieves top-N chunks from the index.
- Answers using only that evidence.
- (Optional) Runs a critic model to check for hallucinations.


## Final report composer

Writes a full markdown research report with:

- Title, Abstract, Introduction
- Findings with subsections
- Risks / Limitations
- Conclusion
- References (URLs)


## CLI options

```
--with-search    Enable live web search + scraping
--max-results    Limit web search result count per subquestion
--quiet          Reduce console logging
--no-critic      Disable the critic pass
```


## Saved outputs

- Final reports saved as markdown under `reports_output/`
- Cleaned page texts under `data/cleaned/`


## Architecture

### High-level components

**GroqLLMClient**  
Thin wrapper around the Groq `/chat/completions` API.

### Agents (`src/agents/`)

- `planner.py` – turns the user query into a research plan  
- `searcher.py` – runs web search and collects raw pages  
- `retriever.py` – builds/queries the local RAG index  
- `answerer.py` – answers each subquestion using top-K chunks  
- `critic.py` – optional faithfulness check  
- `report_composer.py` – composes the final markdown report  

### Entry point

- `src/main.py` – CLI orchestration for the entire pipeline  


## Requirements

- Python 3.10+ (3.11 recommended)
- A Groq API key
- Git


## Setup

### Clone the repo

```bash
git clone https://github.com/<your-username>/atlas-agent.git
cd atlas-agent
```

### Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```


## Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Optional:

```
GROQ_MODEL=llama3-70b-8192
```

Make sure `.env` is in `.gitignore`.


## Usage

### Basic command

```bash
python -m src.main "impact of generative AI on entry-level software engineering roles in India"
```

### With web search enabled

```bash
python -m src.main "impact..." --with-search
```

### Limit web results per subquestion

```bash
python -m src.main "impact..." --with-search --max-results 2
```

### Run quietly

```bash
python -m src.main "impact..." --with-search --quiet
```

### Disable critic

```bash
python -m src.main "impact..." --with-search --no-critic
```

### Multi-word query without quotes

```bash
python -m src.main impact of generative AI on entry-level software engineering roles in India --with-search
```


## Outputs

### Final reports

```
reports_output/
  <slug>-<timestamp>.md
```

### Cleaned text files

```
data/cleaned/
  <hash1>.txt
  <hash2>.txt
```

Each corresponds to a cleaned webpage.


## Running Without Web Search

If `data/cleaned/` contains cleaned text files:

```bash
python -m src.main "your research question"
```

ATLAS will:

- Skip live web search
- Build a retrieval index from the local files
- Answer based only on local evidence


## Project Structure (simplified)

```
atlas-agent/
├─ data/
│  └─ cleaned/
├─ reports_output/
├─ src/
│  ├─ agents/
│  │  ├─ planner.py
│  │  ├─ searcher.py
│  │  ├─ retriever.py
│  │  ├─ answerer.py
│  │  ├─ critic.py
│  │  └─ report_composer.py
│  ├─ llm_client.py
│  └─ main.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Design Notes

### LLM client is centralized

All model calls go through `GroqLLMClient`, making it easy to:

- Swap models  
- Modify parameters  
- Add logging or telemetry  

### Critic is optional

It can be toggled at runtime depending on whether answer verification is needed.

### Aggressive truncation

The composer and answerer truncate long content to avoid context-length issues.
