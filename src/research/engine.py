#!/usr/bin/env python3
"""
ResearchEngine — autonomous multi-turn research with memory.
"""

import sys
from pathlib import Path

# Allow running as script (for CLI testing)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from .memory import Memory, Source
from .search import WebSearch, ImageSearch, ArxivSearch, SearchResult
from .fetchers import WebFetcher, PaperFetcher, FetchResult
from .summarizer import extract_findings_from_source
from .assimilator import Assimilator


@dataclass
class ResearchQuery:
    text: str
    include_images: bool = False
    include_papers: bool = False


@dataclass
class ResearchReport:
    query: str
    session_id: str
    depth: int
    sources_found: int
    sources_fetched: int
    code_blocks: int
    papers: int
    images: int
    findings: list[str]
    sources: list[dict]
    code_snippets: list[str]
    follow_ups: list[str]
    generated_at: str


class ResearchEngine:
    """
    Autonomous research engine.

    1. Takes a research query
    2. Runs initial search(es)
    3. Fetches top results
    4. Generates follow-up queries based on findings
    5. Repeats up to N turns
    6. Produces a structured report
    """

    def __init__(self):
        self.web_search = WebSearch()
        self.image_search = ImageSearch()
        self.arxiv_search = ArxivSearch()
        self.web_fetcher = WebFetcher()
        self.paper_fetcher = PaperFetcher()
        self.memory: Optional[Memory] = None

    # ── Public API ──────────────────────────────────────────────

    def research(
        self,
        query: str,
        max_turns: int = 3,
        include_images: bool = False,
        include_papers: bool = False,
        max_sources_per_turn: int = 5,
        session_id: Optional[str] = None,
    ) -> ResearchReport:
        """
        Run full autonomous research session.

        Returns a ResearchReport with all findings, sources, code, and follow-ups.
        """
        self.memory = Memory(session_id=session_id)
        self.memory.new_session(query)

        print(f"\n\033[1;36m🔬 Research Engine starting\033[0m — query: \"{query}\"")
        print(f"   turns={max_turns}, images={include_images}, papers={include_papers}\n")

        findings: list[str] = []
        all_code: list[str] = []
        all_sources: list[dict] = []
        follow_ups_run: list[str] = []

        for turn in range(1, max_turns + 1):
            print(f"\n\033[1;35m═══ Turn {turn}/{max_turns} ═══\033[0m")

            # Decide query for this turn
            if turn == 1:
                current_query = query
            else:
                current_query = self._generate_follow_up(query, findings)
                if not current_query:
                    print("  → No useful follow-up found, stopping.")
                    break
                follow_ups_run.append(current_query)
                self.memory.add_follow_up(current_query)
                print(f"  → Follow-up query: \"{current_query}\"")

            # ── Run searches ────────────────────────────────────
            web_results = self.web_search.search(current_query, num_results=max_sources_per_turn)
            print(f"  → Web: {len(web_results)} results")

            papers = []
            if include_papers and turn == 1:
                papers = self.arxiv_search.search(query, num_results=3)
                print(f"  → Papers: {len(papers)} results")

            images = []
            if include_images and turn == 1:
                images = self.image_search.search(query, num_results=6)
                print(f"  → Images: {len(images)} results")

            # ── Deduplicate vs memory ────────────────────────────
            fetched_urls = self.memory.get_fetched_urls()
            new_web = [r for r in web_results if r.url not in fetched_urls]
            print(f"  → {len(new_web)} new (not yet fetched)")

            # ── Fetch new sources ─────────────────────────────────
            sources_this_turn = 0
            for r in new_web[:max_sources_per_turn]:
                if sources_this_turn >= max_sources_per_turn:
                    break
                if r.url in fetched_urls:
                    continue

                result = self._fetch_result(r)
                if result.fetch_status == "success" or result.fetch_status == "blocked":
                    sources_this_turn += 1
                    all_sources.append({
                        "title": result.title or r.title,
                        "url": result.url,
                        "snippet": r.snippet,
                        "summary": result.content[:1000] if result.content else r.snippet,
                        "code": result.code_blocks,
                        "status": result.fetch_status,
                        "type": result.source_type,
                    })

                    self.memory.add_source(Source(
                        title=result.title or r.title,
                        url=result.url,
                        snippet=r.snippet,
                        summary=result.content[:1000] if result.content else r.snippet,
                        code_blocks=result.code_blocks,
                        fetch_status=result.fetch_status,
                        source_type=result.source_type,
                        content_length=result.content_length,
                    ))

                    if result.code_blocks:
                        all_code.extend(result.code_blocks)

                    fetched_urls.add(r.url)

            # ── Fetch papers ─────────────────────────────────────
            for paper in papers[:2]:
                result = self._fetch_paper(paper)
                if result.fetch_status == "success":
                    all_sources.append({
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.content[:500],
                        "summary": result.content,
                        "code": [],
                        "status": "success",
                        "type": "paper",
                    })
                    self.memory.add_source(Source(
                        title=result.title,
                        url=result.url,
                        snippet=result.content[:300],
                        summary=result.content,
                        code_blocks=[],
                        fetch_status="success",
                        source_type="paper",
                        content_length=result.content_length,
                    ))

            # ── Synthesize findings for this turn ────────────────
            turn_findings = self._synthesize_findings(web_results, all_sources[-sources_this_turn:])
            if turn_findings:
                findings.extend(turn_findings)
                self.memory.add_findings(turn_findings)

            print(f"  → Turn {turn} complete. Total sources: {len(all_sources)}, code blocks: {len(all_code)}")

        # ── Generate report ─────────────────────────────────────────
        report = ResearchReport(
            query=query,
            session_id=self.memory.session_id,
            depth=max_turns,
            sources_found=len(all_sources),
            sources_fetched=sum(1 for s in all_sources if s["status"] == "success"),
            code_blocks=len(all_code),
            papers=len([s for s in all_sources if s["type"] == "paper"]),
            images=len(images) if "images" in dir() else 0,
            findings=findings,
            sources=all_sources,
            code_snippets=all_code[:20],  # cap at 20
            follow_ups=follow_ups_run,
            generated_at=datetime.utcnow().isoformat(),
        )

        self.memory.close()

        # ── Assimilate findings into Hermes memory + skill registry ──
        assim = Assimilator()
        assim_result = assim.run(report.session_id)
        if assim_result['assimilated']:
            skill_msg = assim_result['skill_slug'] or assim_result['reason']
            print(f"\n🧠 Assimilated — memory={assim_result['memory_updated']}, skill: {skill_msg}")

        return report

    # ── Internals ──────────────────────────────────────────────────

    def _fetch_result(self, result: SearchResult) -> FetchResult:
        """Fetch a web search result."""
        return self.web_fetcher.fetch(result.url)

    def _fetch_paper(self, result: SearchResult) -> FetchResult:
        """Fetch an arXiv paper."""
        return self.paper_fetcher.fetch(result.url)

    def _generate_follow_up(self, original_query: str, findings: list[str]) -> Optional[str]:
        """Generate a targeted follow-up query from findings."""
        if not findings:
            return None

        # Simple heuristic: extract key terms from findings
        # Look for technical terms, missing topics, clarifications needed
        combined = " ".join(findings)

        # Common follow-up patterns
        follow_up_templates = [
            f"{original_query} best practices 2026",
            f"{original_query} implementation example",
            f"{original_query} performance optimization",
            f"{original_query} common pitfalls",
            f"{original_query} TypeScript types",
            f"{original_query} authentication database",
        ]

        # Pick one based on what's missing
        if "example" not in combined.lower() and "code" not in combined.lower():
            return follow_up_templates[1]
        if "performance" not in combined.lower() and "speed" not in combined.lower():
            return follow_up_templates[2]
        if "error" not in combined.lower() and "issue" not in combined.lower():
            return follow_up_templates[3]

        return follow_up_templates[0]

    def _synthesize_findings(self, results: list[SearchResult], new_sources: list[dict]) -> list[str]:
        """Extract real findings from this turn's fetched sources (not snippets)."""
        if not new_sources:
            return []

        all_findings = []
        for source in new_sources:
            # content field is set by the fetcher for successful fetches
            extracted = extract_findings_from_source(source, max_per_source=3)
            all_findings.extend(extracted)

        # Deduplicate across all sources this turn
        seen = set()
        unique = []
        for f in all_findings:
            if f not in seen:
                seen.add(f)
                unique.append(f)

        return unique[:5]  # cap per turn


# ── CLI entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    # Fix path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from research.engine import ResearchEngine

    if len(sys.argv) < 2:
        print("Usage: python engine.py <query> [max_turns] [include_papers:0|1] [include_images:0|1]")
        sys.exit(1)

    query = sys.argv[1]
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    include_papers = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    include_images = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    engine = ResearchEngine()
    report = engine.research(
        query=query,
        max_turns=max_turns,
        include_papers=include_papers,
        include_images=include_images,
    )

    # Print report
    print("\n" + "=" * 60)
    print("\033[1;33m📋 RESEARCH REPORT\033[0m")
    print("=" * 60)
    print(f"\nQuery: {report.query}")
    print(f"Session: {report.session_id}")
    print(f"Depth: {report.depth} turns")
    print(f"Sources: {report.sources_found} found, {report.sources_fetched} fetched")
    print(f"Code blocks: {report.code_blocks}")
    if report.papers:
        print(f"Papers: {report.papers}")
    if report.images:
        print(f"Images: {report.images}")

    if report.findings:
        print(f"\n\033[1;36m📌 Findings:\033[0m")
        for f in report.findings:
            print(f"  • {f}")

    if report.sources:
        print(f"\n\033[1;36m📚 Sources ({len(report.sources)}):\033[0m")
        for i, s in enumerate(report.sources, 1):
            status_icon = "✓" if s["status"] == "success" else "✗"
            print(f"  {status_icon} [{i}] {s['title'][:60]}")
            print(f"      {s['url'][:70]}")

    if report.code_snippets:
        print(f"\n\033[1;36m💻 Code ({len(report.code_snippets)} snippets):\033[0m")
        for i, code in enumerate(report.code_snippets, 1):
            print(f"\n--- Snippet {i} ---")
            print(code[:500] + ("..." if len(code) > 500 else ""))

    if report.follow_ups:
        print(f"\n\033[1;36m🔄 Follow-up queries run:\033[0m")
        for q in report.follow_ups:
            print(f"  • {q}")

    print(f"\n\033[1;32m✅ Report complete.\033[0m {report.sources_found} sources, {report.code_blocks} code blocks.")
    print(f"   Generated at: {report.generated_at}")

    # Save to session file
    session_file = Path.home() / ".hermes" / "research_sessions" / f"{report.session_id}.json"
    print(f"\n📁 Session saved to: {session_file}")
