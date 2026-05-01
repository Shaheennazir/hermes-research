# hermes-research

Autonomous multi-turn web research engine with memory, ranking, deduplication, code extraction, and MCP server — built for [Hermes Agent](https://github.com/nousresearch/hermes-agent).

Research sessions accumulate across turns, with findings feeding back into a memory layer that informs future sessions. The MCP server exposes the engine as tools for any MCP-capable agent.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  research_web(query, context_hint)   ← MCP tool call         │
│                      ↓                                      │
│  Assimilator.get_prior_context()                            │
│    → inject_prior_context()  ← memory.jsonl lookup          │
│                      ↓                                      │
│  ResearchEngine.run()  ← multi-turn loop                    │
│    ├── SearchAgent   → mmx search (web, images, arxiv)     │
│    ├── FetchAgent   → fetch + parse + extract code        │
│    └── Synthesizer  → combine findings across turns        │
│                      ↓                                      │
│  Assimilator.run()  ← post-session                         │
│    ├── Summarizer   → MemoryEntry with quality score      │
│    ├── Memory       → memory.jsonl update                  │
│    └── Skills       → auto-skill creation if quality gate │
└─────────────────────────────────────────────────────────────┘
```

**10 modules, ~2,300 lines:**

| Module | Purpose |
|---|---|
| `engine.py` | Core ResearchEngine — orchestrates multi-turn loop |
| `search.py` | WebSearch (mmx CLI), ImageSearch, ArxivSearch |
| `fetchers.py` | WebFetcher, PaperFetcher with content_length tracking |
| `summarizer.py` | Session → MemoryEntry, quality scoring (Signals A–D) |
| `memory.py` | Session memory, source tracking, deduplication |
| `topic_similarity.py` | Jaccard matching + recency decay for prior context |
| `skills_registry.py` | Auto-skill creation, separate from main skill namespace |
| `assimilator.py` | Orchestrates summarize → memory → skill lifecycle |
| `mcp_server.py` | MCP stdio server exposing `research_web`, `research_session`, `research_clear` |

---

## Installation

```bash
# Clone
git clone https://github.com/Shaheennazir/hermes-research.git
cd hermes-research

# Install dependencies
pip install -e .
```

**Requirements:** Python 3.11+, `bs4`, `lxml`, `requests`, `mcp`

---

## Quick Start

### 1. Authenticate mmx (search backend)

```bash
mmx auth login --api-key sk-your-key-here
```

Search works without env vars — credentials are stored in `~/.mmx/config.json`.

### 2. Run a research session directly

```python
from research import ResearchEngine

engine = ResearchEngine()
result = engine.run("Next.js Server Actions best practices", turns=3)

print(result['report'])
# → Multi-turn synthesized report with sources + code blocks
```

### 3. Use as MCP tools

Add to your `config.yaml` (or `settings.json` for Claude Code):

```yaml
mcp_servers:
  research:
    command: python
    args: [/path/to/hermes-research/src/mcp_server.py]
```

Restart your agent. Three tools become available:

- **`research_web(query, turns?, context_hint?)`** — run a full multi-turn session
- **`research_session(session_id)`** — retrieve a past session by ID
- **`research_clear()`** — wipe session history

### 4. Integrate prior context into your agent prompt

The MCP server handles this automatically — every `research_web` call injects relevant prior context before the engine runs. No extra setup needed.

---

## Configuration

| Env / Config | Default | Description |
|---|---|---|
| `HERMES_RESEARCH_DIR` | `~/.hermes/research/` | Base data directory |
| `HERMES_RESEARCH_SESSIONS` | `~/.hermes/research_sessions/` | Session JSON files |
| `HERMES_RESEARCH_SKILLS` | `~/.hermes/research_skills/` | Auto-generated skills |
| `HERMES_MEMORY_PATH` | `~/.hermes/memory.jsonl` | Hermes memory file |

Set any of these to move the data directory.

---

## How It Works

### Multi-turn loop

The engine runs N turns. Each turn:
1. Generate search queries from the topic + prior findings
2. Execute searches in parallel (web + images + arxiv)
3. Fetch and parse results
4. Extract code blocks, facts, and links
5. Synthesize findings — next turn's queries are informed by this turn's results

### Quality scoring (Signals A–D)

Every session is scored on 4 signals before storage:

| Signal | Weight | What it measures |
|---|---|---|
| A | 0.20 | Source diversity (Jaccard on domains) |
| B | 0.15 | Finding specificity (version numbers, method signatures, HTTP codes) |
| C | 0.30 | Code relevance (snippet length + keyword overlap with query) |
| D | 0.35 | Fetch success rate (content_length > 0 at fetcher level) |

Floor: ≥3 successful sources AND ≥2 code blocks. Below floor → score = 0.

### Prior context injection

Before each new session, the system looks up relevant entries from `memory.jsonl` using Jaccard keyword similarity × recency multiplier:

```
adjusted_score = jaccard × (0.5 ^ (days_old / 30))
```

Entries older than 90 days contribute at most 12.5% of their raw similarity. Quality filter: only entries with score ≥ 0.2 are considered.

### Auto-skill creation

Sessions passing quality gate (≥0.3 weighted, ≥3 keywords, ≥3 sources) and passing the Jaccard dedup check (similarity < 0.5 vs existing skills) automatically generate a research skill in `~/.hermes/research_skills/`. These are separate from the main skill namespace until explicitly promoted.

---

## ⚠️ How NOT to Use This

**Don't use for real-time data needs.**
The engine is built for research accumulation — it's slow (multi-turn loops), not a live API. For stock prices, weather, sports scores, use a direct API call.

**Don't use as a general-purpose search engine replacement.**
Sessions take 30–120 seconds. The MCP server is designed for agents doing deep research on a topic, not ad-hoc single queries. If you want a one-shot web search, use `mmx search` directly.

**Don't expect the summarizer to have semantic understanding.**
The summarizer is rule-based (regex + heuristics). It extracts findings based on pattern matching — version numbers, method signatures, sentences with technical verbs. It does not synthesize new insights. Generic boilerplate from engine output will be penalized by Signal B but won't be filtered out completely. The summarizer is the weakest component and will be LLM-powered in a future release.

**Don't expect memory.jsonl to stay clean without quality input.**
If source pages are paywalled, blocked, or low-content, the engine will store weak findings. A weak session produces a weak MemoryEntry that pollutes future context injection. The quality gate catches obviously bad sessions but isn't a substitute for good sources.

**Don't run the MCP server without authentication.**
`mmx auth login` is required. Without it, all searches fail silently and sessions return empty results.

**Don't use memory.jsonl as a permanent knowledge base.**
It's a research memory for the agent, not a curated KB. Entries are auto-generated with no manual review. Treat it as a cache of research sessions, not a source of truth.

---

## API Reference

### `ResearchEngine.run(query, turns=2, depth=0, context_hint="")`

Run a full research session.

**Returns:**
```python
{
    'session_id': 'session_1234567890',
    'query': str,
    'turns': int,
    'sources': list[dict],       # fetched + parsed sources
    'findings': list[str],      # synthesized findings per turn
    'report': str,               # full markdown report
    'assimilated': bool,        # whether it was stored in memory
}
```

### `research_web(query, turns=2, context_hint="")` (MCP tool)

Full session with automatic prior context injection and assimilation.

### `research_session(session_id)` (MCP tool)

Retrieve a past session from disk. Returns the full session JSON.

### `research_clear()` (MCP tool)

Delete all session files and assimilation markers. Does NOT touch memory.jsonl.

---

## File Structure

```
hermes-research/
├── README.md
├── pyproject.toml
├── LICENSE
├── src/
│   └── research/          # Package
│       ├── __init__.py
│       ├── engine.py      # ResearchEngine (main loop)
│       ├── search.py      # mmx, image, arxiv backends
│       ├── fetchers.py    # WebFetcher, PaperFetcher
│       ├── summarizer.py  # Session → MemoryEntry
│       ├── memory.py      # Session memory + dedup
│       ├── topic_similarity.py  # Jaccard + recency
│       ├── skills_registry.py   # Auto-skill creation
│       ├── assimilator.py # Full lifecycle orchestrator
│       └── mcp_server.py  # MCP stdio server
├── tests/
│   └── test_*.py
└── docs/
    └── architecture.md
```
