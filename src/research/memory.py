#!/usr/bin/env python3
"""
Research Session Memory — tracks findings across multi-turn research.
"""

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Source:
    title: str
    url: str
    snippet: str
    summary: str
    code_blocks: list[str] = field(default_factory=list)
    fetch_status: str = "success"  # success | blocked | failed
    source_type: str = "web"  # web | image | paper
    relevance_score: float = 0.0
    fetched_at: str = ""
    content_length: int = 0  # raw char count (Signal D input)


@dataclass
class ResearchSession:
    id: str
    query: str
    created_at: str
    updated_at: str
    sources: list[Source] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    research_depth: int = 1
    total_code_blocks: int = 0


class Memory:
    """
    Stores research session state to:
    - Avoid re-fetching the same URLs across turns
    - Track what's already been found
    - Attach context to new searches
    """

    SESSION_DIR = Path.home() / ".hermes" / "research_sessions"

    def __init__(self, session_id: Optional[str] = None):
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_path = self.SESSION_DIR / f"{self.session_id}.json"
        self.session: Optional[ResearchSession] = None
        self._load()

    def _load(self):
        if self.session_path.exists():
            try:
                data = json.loads(self.session_path.read_text())
                self.session = ResearchSession(**data)
            except Exception:
                self.session = None

    def _save(self):
        if self.session:
            self.session.updated_at = datetime.utcnow().isoformat()
            self.session_path.write_text(json.dumps(asdict(self.session), indent=2))

    def new_session(self, query: str) -> ResearchSession:
        now = datetime.utcnow().isoformat()
        self.session = ResearchSession(
            id=self.session_id,
            query=query,
            created_at=now,
            updated_at=now,
            sources=[],
            findings=[],
            follow_up_queries=[],
            research_depth=1,
            total_code_blocks=0,
        )
        self._save()
        return self.session

    def add_source(self, source: Source):
        if not self.session:
            return
        # Deduplicate by URL
        existing = {s.url for s in self.session.sources}
        if source.url not in existing:
            self.session.sources.append(source)
            self.session.total_code_blocks += len(source.code_blocks)
        self._save()

    def add_findings(self, findings: list[str]):
        if not self.session:
            return
        for f in findings:
            if f not in self.session.findings:
                self.session.findings.append(f)
        self._save()

    def add_follow_up(self, query: str):
        if not self.session:
            return
        if query not in self.session.follow_up_queries:
            self.session.follow_up_queries.append(query)
            self.session.research_depth += 1
        self._save()

    def get_fetched_urls(self) -> set[str]:
        if not self.session:
            return set()
        return {s.url for s in self.session.sources}

    def get_previous_findings(self) -> list[str]:
        if not self.session:
            return []
        return self.session.findings

    def get_source_count(self) -> int:
        if not self.session:
            return 0
        return len(self.session.sources)

    def get_session_summary(self) -> dict:
        if not self.session:
            return {}
        return {
            "query": self.session.query,
            "depth": self.session.research_depth,
            "sources": len(self.session.sources),
            "code_blocks": self.session.total_code_blocks,
            "findings": len(self.session.findings),
            "follow_ups": len(self.session.follow_up_queries),
        }

    def close(self):
        self.session = None
