#!/usr/bin/env python3
"""
Skills Registry — separate store for auto-generated research skills.

Stores MemoryEntries as individual skill files in:
  ~/.hermes/research_skills/

Each skill is a JSON file named after a slug of the topic.
Registry index maps topic slugs to file paths for fast lookup.
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from .summarizer import MemoryEntry
from .topic_similarity import topic_similarity, should_create_skill


REGISTRY_DIR = Path.home() / '.hermes' / 'research_skills'
INDEX_FILE = REGISTRY_DIR / 'index.json'


class SkillsRegistry:
    """
    Separate storage for auto-generated research skills.
    Does NOT pollute ~/.hermes/skills/ until quality is proven.
    """

    def __init__(self):
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ── Index management ────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        if INDEX_FILE.exists():
            try:
                return json.loads(INDEX_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_index(self):
        INDEX_FILE.write_text(json.dumps(self._index, indent=2))

    def _slugify(self, topic: str) -> str:
        """Convert topic string to a safe filename slug."""
        import re
        # Lowercase, replace non-alphanumeric with underscore, limit length
        slug = re.sub(r'[^a-z0-9]+', '_', topic.lower())
        slug = slug.strip('_')[:60]
        return slug or 'unnamed'

    # ── CRUD ───────────────────────────────────────────────

    def list_skills(self) -> list[dict]:
        """Return list of all skills (summary info, no full content)."""
        return list(self._index.values())

    def get_skill(self, slug: str) -> Optional[MemoryEntry]:
        """Load a skill by slug."""
        entry = self._index.get(slug)
        if not entry:
            return None
        path = REGISTRY_DIR / entry['filename']
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return MemoryEntry(**data)

    def get_all_skills(self) -> list[MemoryEntry]:
        """Load all skills as MemoryEntry objects."""
        skills = []
        for slug in self._index:
            skill = self.get_skill(slug)
            if skill:
                skills.append(skill)
        return skills

    def add_skill(self, entry: MemoryEntry) -> str:
        """
        Add a new research skill.
        Returns the slug it was saved under.
        """
        # Generate unique slug
        base_slug = self._slugify(entry.topic)
        slug = base_slug
        counter = 1
        while slug in self._index:
            slug = f"{base_slug}_{counter}"
            counter += 1

        filename = f"{slug}.json"
        filepath = REGISTRY_DIR / filename

        # Write skill file
        filepath.write_text(json.dumps(entry.to_dict(), indent=2))

        # Update index
        self._index[slug] = {
            'slug': slug,
            'filename': filename,
            'topic': entry.topic,
            'topic_keywords': entry.topic_keywords[:8],
            'assimilated_at': entry.assimilated_at,
            'quality_score': entry.quality_score,
            'depth': entry.depth,
            'session_count': len(entry.session_ids),
        }
        self._save_index()

        return slug

    def merge_into_skill(
        self,
        existing_slug: str,
        new_entry: MemoryEntry,
    ) -> bool:
        """
        Merge new session data into an existing skill.
        Updates keywords, findings, sources, code patterns.
        Returns True if merged, False if skill not found.
        """
        skill = self.get_skill(existing_slug)
        if not skill:
            return False

        # Merge keywords (union, dedupe)
        existing_kw = set(skill.topic_keywords)
        existing_kw.update(new_entry.topic_keywords)
        skill.topic_keywords = list(existing_kw)[:30]

        # Merge findings (dedupe by similarity)
        existing_findings = skill.key_findings
        for f in new_entry.key_findings:
            is_dup = False
            for ef in existing_findings:
                # Simple overlap check
                words_f = set(f.lower().split())
                words_ef = set(ef.lower().split())
                overlap = len(words_f & words_ef) / max(len(words_f), len(words_ef))
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                existing_findings.append(f)
        skill.key_findings = existing_findings[:15]

        # Merge important sources (dedupe by URL)
        existing_urls = {s['url'] for s in skill.important_sources}
        for s in new_entry.important_sources:
            if s['url'] not in existing_urls:
                skill.important_sources.append(s)
                existing_urls.add(s['url'])
        skill.important_sources = skill.important_sources[:15]

        # Merge code patterns
        existing_code = set(skill.code_patterns)
        for c in new_entry.code_patterns:
            existing_code.add(c)
        skill.code_patterns = list(existing_code)[:10]

        # Update metadata
        skill.session_ids.append(new_entry.session_ids[0])
        skill.session_ids = skill.session_ids[:20]  # cap
        skill.depth = max(skill.depth, new_entry.depth)
        skill.quality_score = max(skill.quality_score, new_entry.quality_score)
        skill.assimilated_at = datetime.utcnow().isoformat()

        # Save
        filepath = REGISTRY_DIR / self._index[existing_slug]['filename']
        filepath.write_text(json.dumps(skill.to_dict(), indent=2))

        # Update index
        self._index[existing_slug]['assimilated_at'] = skill.assimilated_at
        self._index[existing_slug]['quality_score'] = skill.quality_score
        self._index[existing_slug]['session_count'] = len(skill.session_ids)
        self._save_index()

        return True

    def delete_skill(self, slug: str) -> bool:
        """Delete a skill by slug."""
        if slug not in self._index:
            return False
        entry = self._index[slug]
        filepath = REGISTRY_DIR / entry['filename']
        if filepath.exists():
            filepath.unlink()
        del self._index[slug]
        self._save_index()
        return True

    # ── Query ──────────────────────────────────────────────

    def find_similar(self, entry: MemoryEntry, threshold: float = 0.5) -> list[tuple[str, float]]:
        """
        Find skills similar to a MemoryEntry.
        Returns list of (slug, similarity_score) sorted descending.
        """
        all_skills = self.get_all_skills()
        matches = []
        for skill in all_skills:
            sim = topic_similarity(entry, skill)
            if sim >= threshold:
                matches.append((skill.topic_keywords, sim))  # placeholder
        return []

    # ── CLI ─────────────────────────────────────────────────

    def dump(self):
        """Print all skills summary."""
        if not self._index:
            print("No research skills yet.")
            return
        for slug, info in self._index.items():
            print(f"\n[{slug}]")
            print(f"  Topic: {info['topic']}")
            print(f"  Keywords: {', '.join(info['topic_keywords'])}")
            print(f"  Quality: {info['quality_score']}")
            print(f"  Sessions: {info['session_count']}")
            print(f"  Last updated: {info['assimilated_at']}")


if __name__ == '__main__':
    import sys
    reg = SkillsRegistry()

    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        reg.dump()
    else:
        print(f"Registry at: {REGISTRY_DIR}")
        print(f"Skills count: {len(reg.list_skills())}")
