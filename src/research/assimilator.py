#!/usr/bin/env python3
"""
Assimilator — orchestrates the post-research assimilation loop.

Triggered by engine.py after each research session completes.
Coordinates:
  1. Summarize session → MemoryEntry
  2. Update Hermes memory with key findings
  3. Evaluate skill creation (separate registry)
  4. Mark session as assimilated

Keeps the research layer owning the lifecycle — no Hermes internals.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .summarizer import Summarizer, MemoryEntry
from .topic_similarity import should_create_skill, find_prior_research, inject_prior_context
from .skills_registry import SkillsRegistry

# Hermes memory tool path (we write directly to the memory store)
HERMES_MEMORY_FILE = Path.home() / '.hermes' / 'memory.jsonl'


class Assimilator:
    """
    Orchestrates post-research assimilation.

    Call after engine.research() returns a report.
    """

    def __init__(self):
        self.summarizer = Summarizer()
        self.skills = SkillsRegistry()

    def run(self, session_id: str) -> dict:
        """
        Full assimilation loop for a completed session.

        Returns:
            dict with keys:
              - assimilated: bool
              - memory_updated: bool
              - skill_created: bool
              - skill_slug: str or None
              - reason: str
        """
        result = {
            'session_id': session_id,
            'assimilated': False,
            'memory_updated': False,
            'skill_created': False,
            'skill_slug': None,
            'reason': '',
            'timestamp': datetime.utcnow().isoformat(),
        }

        # 1. Check if already assimilated
        if self._is_assimilated(session_id):
            result['reason'] = 'already assimilated'
            return result

        # 2. Summarize session → MemoryEntry
        try:
            entry = self.summarizer.summarize(session_id)
        except Exception as e:
            result['reason'] = f'summarization failed: {e}'
            return result

        entry.assimilated_at = result['timestamp']

        # 3. Update Hermes memory with key findings
        try:
            self._update_hermes_memory(entry)
            result['memory_updated'] = True
        except Exception as e:
            result['reason'] = f'memory update failed: {e}'

        # 4. Evaluate skill creation
        try:
            existing_skills = self.skills.get_all_skills()
            should_create, reason = should_create_skill(entry, existing_skills)

            if should_create:
                # Check if similar skill exists to merge into
                similar = self._find_similar_skill(entry, existing_skills)
                if similar:
                    merged = self.skills.merge_into_skill(similar, entry)
                    if merged:
                        result['skill_created'] = False
                        result['skill_slug'] = similar
                        result['reason'] = f'merged into existing skill: {similar}'
                    else:
                        result['reason'] = f'merge failed for: {similar}'
                else:
                    slug = self.skills.add_skill(entry)
                    result['skill_created'] = True
                    result['skill_slug'] = slug
                    result['reason'] = f'new skill created: {slug}'
            else:
                result['reason'] = reason or 'quality/similarity gate not met'

        except Exception as e:
            result['reason'] = f'skill evaluation failed: {e}'

        # 5. Mark session as assimilated
        self._mark_assimilated(session_id)

        result['assimilated'] = True
        return result

    # ── Hermes memory ─────────────────────────────────────

    def _update_hermes_memory(self, entry: MemoryEntry):
        """
        Append a memory entry to Hermes's persistent memory store.
        Writes to ~/.hermes/memory.jsonl in the format Hermes expects.
        """
        memory_record = {
            'type': 'research_finding',
            'topic': entry.topic,
            'keywords': entry.topic_keywords,
            'findings': entry.key_findings,
            'sources': entry.important_sources,
            'session_ids': entry.session_ids,
            'quality': entry.quality_score,
            'assimilated_at': entry.assimilated_at,
        }

        HERMES_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Append as JSONL line
        with open(HERMES_MEMORY_FILE, 'a') as f:
            f.write(json.dumps(memory_record) + '\n')

    # ── Session tracking ──────────────────────────────────

    def _assimilation_marker_path(self, session_id: str) -> Path:
        return Path.home() / '.hermes' / 'research_sessions' / f'{session_id}_assimilated.json'

    def _is_assimilated(self, session_id: str) -> bool:
        return self._assimilation_marker_path(session_id).exists()

    def _mark_assimilated(self, session_id: str):
        marker = self._assimilation_marker_path(session_id)
        marker.write_text(json.dumps({
            'session_id': session_id,
            'assimilated_at': datetime.utcnow().isoformat(),
        }))

    # ── Skill helpers ─────────────────────────────────────

    def _find_similar_skill(
        self,
        entry: MemoryEntry,
        existing_skills: list[MemoryEntry],
    ) -> Optional[str]:
        """
        Find an existing skill slug that's similar enough to merge into.
        Uses Jaccard similarity on keywords.
        """
        for skill in existing_skills:
            from .topic_similarity import topic_similarity
            sim = topic_similarity(entry, skill)
            if sim >= 0.5:  # SKILL_CREATION_THRESHOLD
                # Return the slug for this skill
                for slug, info in self.skills._index.items():
                    if info['topic'] == skill.topic:
                        return slug
        return None

    # ── Prior research for new sessions ───────────────────

    @staticmethod
    def get_prior_context(topic_keywords: list[str]) -> str:
        """
        Get compressed prior research context for a new session.
        Called at session start, not post-research.
        """
        # Load all memory entries
        memory_entries = Assimilator._load_memory_entries()
        prior = find_prior_research(topic_keywords, memory_entries)
        return inject_prior_context(prior)

    @staticmethod
    def _load_memory_entries() -> list[MemoryEntry]:
        """Load all research findings from Hermes memory."""
        if not HERMES_MEMORY_FILE.exists():
            return []

        entries = []
        for line in HERMES_MEMORY_FILE.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get('type') == 'research_finding':
                    # Reconstruct MemoryEntry from stored record
                    entries.append(MemoryEntry(
                        topic=record.get('topic', ''),
                        topic_keywords=record.get('keywords', []),
                        key_findings=record.get('findings', []),
                        important_sources=record.get('sources', []),
                        code_patterns=record.get('code_patterns', []),
                        session_ids=record.get('session_ids', []),
                        assimilated_at=record.get('assimilated_at', ''),
                        depth=record.get('depth', 1),
                        quality_score=record.get('quality', 0.0),
                    ))
            except Exception:
                continue
        return entries


# ── CLI ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python assimilator.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    assim = Assimilator()
    result = assim.run(session_id)

    print(json.dumps(result, indent=2))

    if result['skill_created']:
        print(f"\n✅ New skill: {result['skill_slug']}")
    elif result['skill_slug']:
        print(f"\n🔄 Merged into: {result['skill_slug']}")
    else:
        print(f"\n⏭️  No skill: {result['reason']}")
