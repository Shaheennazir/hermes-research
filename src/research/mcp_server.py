#!/usr/bin/env python3
"""
Research MCP Server — exposes ResearchEngine as MCP tools.
Connects via stdio to any MCP-compatible client (e.g. Claude Code, Hermes).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from research.engine import ResearchEngine, ResearchReport
from research.memory import Memory
from research.assimilator import Assimilator
from research.summarizer import extract_keywords


SERVER_NAME = "research"
server = Server(SERVER_NAME)


# ── Tool Definitions ────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="research_web",
            description=(
                "Run autonomous web research on any topic. "
                "Performs multi-turn search + content fetching + code extraction. "
                "Returns a full structured report with sources, summaries, code snippets, and follow-up queries used. "
                "Use when you need deep research on a technical topic, comparison, or comprehensive survey of sources."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research query/topic",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Number of research turns (default: 3, max: 5)",
                        "default": 3,
                    },
                    "include_papers": {
                        "type": "boolean",
                        "description": "Include arXiv academic paper results (default: false)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="research_session",
            description="Get current or past research session summary by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional — returns current if omitted)",
                    },
                },
            },
        ),
        Tool(
            name="research_clear",
            description="Clear all research session files.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ── Tool Handlers ──────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    engine = ResearchEngine()

    if name == "research_web":
        query = arguments["query"]
        max_turns = min(arguments.get("max_turns", 3), 5)
        include_papers = arguments.get("include_papers", False)

        # ── Prior context injection ───────────────────────────
        # Load related prior research before starting new session
        query_keywords = extract_keywords(query)
        prior_context = Assimilator.get_prior_context(query_keywords)

        # If we have prior research, prepend it to the query as context
        if prior_context:
            enriched_query = f"{prior_context}\n\n---\n\n## New Research Query\n\n{query}"
        else:
            enriched_query = query
        # ────────────────────────────────────────────────────

        report = engine.research(
            query=enriched_query,
            max_turns=max_turns,
            include_papers=include_papers,
        )

        return [_report_to_text(report)]

    elif name == "research_session":
        session_id = arguments.get("session_id")
        mem = Memory(session_id=session_id) if session_id else Memory()
        if not mem.session:
            return [TextContent(type="text", text="No active research session found.")]
        summary = mem.get_session_summary()
        mem.close()
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]

    elif name == "research_clear":
        sessions_dir = Path.home() / ".hermes" / "research_sessions"
        if sessions_dir.exists():
            for f in sessions_dir.glob("*.json"):
                f.unlink()
        return [TextContent(type="text", text="All research sessions cleared.")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def _report_to_text(report: ResearchReport) -> TextContent:
    """Convert a ResearchReport to readable markdown."""
    lines = [
        f"# Research Report: {report.query}",
        f"",
        f"**Session:** `{report.session_id}`  **Depth:** {report.depth} turns",
        f"**Sources:** {report.sources_found} found, {report.sources_fetched} fetched",
        f"**Code blocks:** {report.code_blocks}",
    ]

    if report.papers:
        lines.append(f"**Papers:** {report.papers}")
    if report.images:
        lines.append(f"**Images:** {report.images}")

    if report.findings:
        lines += ["", "## Findings"]
        for f in report.findings:
            lines.append(f"- {f}")

    if report.sources:
        lines += ["", "## Sources"]
        for i, s in enumerate(report.sources, 1):
            icon = "✓" if s["status"] == "success" else "✗"
            lines.append(f"{icon} [{i}] **{s['title']}**")
            lines.append(f"   {s['url']}")
            if s["summary"]:
                snippet = s["summary"][:200].replace("\n", " ")
                lines.append(f"   _{snippet}..._")

    if report.code_snippets:
        lines += ["", "## Code Snippets"]
        for i, code in enumerate(report.code_snippets, 1):
            preview = code[:600].strip()
            lines.append(f"``````")
            lines.append(f"--- Snippet {i} ---")
            lines.append(preview)
            lines.append("``````")

    if report.follow_ups:
        lines += ["", "## Follow-up Queries Run"]
        for q in report.follow_ups:
            lines.append(f"- `{q}`")

    lines += ["", f"_Report generated: {report.generated_at}_"]

    return TextContent(type="text", text="\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(
            read_stream,
            write_stream,
            init_options,
            raise_exceptions=True,
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
