# Architecture — hermes-research

> How the engine works, module by module.

---

## Overview

The system has three layers:

```
┌─────────────────────────────────────────────────┐
│  MCP Server  (mcp_server.py)                    │
│  → research_web, research_session, research_clear│
└────────────────────┬────────────────────────────┘
                     │ (prior context, assimilation trigger)
┌────────────────────▼────────────────────────────┐
│  ResearchEngine  (engine.py)                   │
│  → multi-turn search / fetch / synthesize loop │
└────────────────────┬────────────────────────────┘
                     │ (session data)
┌────────────────────▼────────────────────────────┐
│  Assimilator  (assimilator.py)                 │
│  → Summarize → Memory → Skills                 │
└─────────────────────────────────────────────────┘
```

---

## Module Map

### `search.py` — Search Backends

| Class | Backend | Notes |
|---|---|---|
| `WebSearch` | `mmx search` | CLI, auth via `~/.mmx/config.json` |
| `ImageSearch` | `mmx search --image` | Returns image URLs only |
| `ArxivSearch` | arXiv.org API | Direct HTTP, no auth |

All search methods return a list of `dict` with at minimum `url`, `title`, `snippet`.

### `fetchers.py` — Content Fetching

| Class | Handles | Auth |
|---|---|---|
| `WebFetcher` | HTTP/HTTPS | User-Agent rotation, retry on 429/503 |
| `PaperFetcher` | arXiv PDFs | Direct download, text extraction |

Both return `FetchResult(native@dataclass)`:
```python
@dataclass
class FetchResult:
    url: str
    status: str           # 'success' | 'blocked' | 'failed'
    title: str
    snippet: str
    summary: str
    code_blocks: list[str]
    links: list[str]
    content_length: int   # bytes, set at fetcher level (Signal D input)
```

### `engine.py` — ResearchEngine

The orchestrator. `run()` executes N turns:

```
Turn 1:
  query_agent.generate_queries() → search_queries
  parallel: search(query) for each query
  parallel: fetch(urls) for each result
  extract: code_blocks, facts, links from each page
  synthesize: combine into findings

Turn 2+:
  Same flow, but query generation informed by prior findings
  context_hint is prepended to synthesis

Post-loop:
  write session JSON to disk
  Assimilator.run()  ← triggered here
```

Key: `context_hint` from MCP tool → injected into the turn 1 synthesis prompt.

### `summarizer.py` — What Gets Stored

`summarize(session_id)` → `MemoryEntry`:

```python
@dataclass
class MemoryEntry:
    topic: str
    topic_keywords: list[str]    # from TECH_TERMS regex + CamelCase + dotted
    key_findings: list[str]       # engine findings + regex-extracted facts
    important_sources: list[dict] # canonical URLs with titles
    code_patterns: list[str]      # distinct snippets, deduped
    session_ids: list[str]
    assimilated_at: str           # ISO timestamp
    depth: int
    quality_score: float          # weighted 0-1
    quality_breakdown: dict      # Signal A/B/C/D
    quality_gate: tuple[bool, str] # (passed, reason)
```

Quality scoring:

| Signal | Weight | Input | How |
|---|---|---|---|
| A | 0.20 | sources | Jaccard of domain sets |
| B | 0.15 | findings | % with specificity anchors |
| C | 0.30 | sources + query | snippet length + keyword overlap |
| D | 0.35 | sources | fetch success rate at fetcher level |

**Signal B specificity anchors:**
- Version numbers: `v?\d+\.\d+(?:\.\d+)?`
- Method signatures: `NAME(...)`
- HTTP codes: `[1-5]\d{2}`
- Timeouts/ports: `timeout\s*[=:]\s*\d+`
- Named constants: `MAX_|DEFAULT_|ERROR_`

**Floor:** ≥3 successful sources AND ≥2 code blocks. Below floor → score = 0.

### `topic_similarity.py` — Prior Context

Two functions:

- `find_prior_research()` — Jaccard keyword similarity × recency decay
- `inject_prior_context()` — compresses MemoryEntries into readable text block

Recency formula: `adjusted = raw_jaccard × (0.5 ^ (days_old / 30))`

At 90+ days, contribution capped at 12.5% of raw similarity.

### `skills_registry.py` — Auto-Skill Creation

Three-layer gate (all must pass):
1. Quality score ≥ 0.3
2. Keyword count ≥ 3
3. Jaccard similarity < 0.5 vs any existing skill

Skills stored in `~/.hermes/research_skills/` (separate from main skill namespace).
Each skill is a JSON file with `SKILL.md`-compatible frontmatter.

### `assimilator.py` — Lifecycle Orchestrator

```
Assimilator.run(session_id):
  1. check if already assimilated
  2. Summarizer.summarize() → MemoryEntry
  3. Memory.update() → append to memory.jsonl
  4. should_create_skill() → if yes, SkillsRegistry.create()
  5. write _assimilated marker
```

Static method: `Assimilator.get_prior_context(query_keywords)` → context block for pre-session injection.

---

## Data Paths

```
~/.hermes/                          # Base (configurable via env)
├── research_sessions/              # Raw session JSON
│   ├── session_1234567890.json
│   └── session_1234567890_assimilated.json
├── research_skills/                # Auto-generated skills
│   └── stripe_webhook_python_best_practices/
│       └── skill.json
└── memory.jsonl                    # Accumulated MemoryEntries (appended)
```

---

## Signal D: Why It's Measured at Fetcher Level

`content_length` is set in `fetchers.py` before any summarization runs. This prevents the summarizer from gaming its own scoring input — the signal is measured independently of where it's used.

---

## Recency Decay: Why Exponential

Linear decay (e.g., subtracting 0.01 per day) means a 90-day-old entry still scores 0.1 on a 0.3 similarity — enough to inject. Exponential decay with a 30-day half-life gives 0.125 at 90 days: low enough that an entry needs genuine relevance (high Jaccard) to still qualify.

Tiebreaking: when two entries have the same adjusted score, the fresher one ranks higher. This prevents fresh-but-loosely-related entries from being buried by old-but-precisely-matched ones.
