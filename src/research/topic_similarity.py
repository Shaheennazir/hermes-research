#!/usr/bin/env python3
"""
Topic Similarity — Jaccard keyword matching for research sessions and skills.

Provides:
- Jaccard similarity between keyword sets
- Topic similarity between MemoryEntries
- Skill deduplication check
- Prior research injection query
"""

from typing import Optional
from datetime import datetime, timezone
from .summarizer import MemoryEntry


# ── Recency decay ────────────────────────────────────────────

# Half-life of 30 days: an entry's relevance is halved every 30 days.
# adjusted_score = jaccard * (0.5 ** (days_old / 30))
RECENCY_DECAY_CONSTANT = 30  # days for half-life
RECENCY_MAX_AGE = 90         # cap decay at 90 days (score ≈ 12.5% of original)


def _recency_multiplier(assimilated_at: str) -> float:
    """
    Exponential decay: score halves every 30 days.
    Returns 0.5 to 1.0 (fresh entries score highest).
    """
    try:
        then = datetime.fromisoformat(assimilated_at.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return 0.5  # treat malformed timestamps as old

    now = datetime.now(timezone.utc)
    # Ensure both are timezone-aware for comparison
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)

    delta = (now - then).total_seconds()
    days_old = max(0, delta / 86400)

    # Cap at RECENCY_MAX_AGE days
    capped_days = min(days_old, RECENCY_MAX_AGE)
    return 0.5 ** (capped_days / RECENCY_DECAY_CONSTANT)


# ── Jaccard similarity ────────────────────────────────────────

def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """
    Jaccard coefficient: |A ∩ B| / |A ∪ B|
    Returns 0.0 to 1.0
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def keyword_similarity(keywords_a: list[str], keywords_b: list[str]) -> float:
    """Jaccard on keyword lists."""
    set_a = set(t.lower().strip() for t in keywords_a)
    set_b = set(t.lower().strip() for t in keywords_b)
    return jaccard(set_a, set_b)


# ── Topic similarity ─────────────────────────────────────────

def topic_similarity(entry_a: MemoryEntry, entry_b: MemoryEntry) -> float:
    """
    Compare two MemoryEntries by topic similarity.
    Combines Jaccard on keywords + exact topic string similarity.
    """
    kw_sim = keyword_similarity(entry_a.topic_keywords, entry_b.topic_keywords)

    # Also compare cleaned topic strings for exact matches
    topic_a_clean = ' '.join(entry_a.topic.lower().split())
    topic_b_clean = ' '.join(entry_b.topic.lower().split())

    # Substring / superstring check
    string_sim = 0.0
    if topic_a_clean == topic_b_clean:
        string_sim = 1.0
    elif topic_a_clean in topic_b_clean or topic_b_clean in topic_a_clean:
        string_sim = 0.7

    # Combined: weighted average
    return max(kw_sim, string_sim)


# ── Prior research injection ──────────────────────────────────

def find_prior_research(
    new_topic_keywords: list[str],
    memory_entries: list[MemoryEntry],
    threshold: float = 0.3,
) -> list[MemoryEntry]:
    """
    Find memory entries relevant to a new research topic.

    Lower threshold (0.3) for injection — cast a wider net.
    Filters out entries with quality_score < 0.2 to avoid injecting noise.

    Recency weighting: exponential decay (half-life 30 days). At 90+ days,
    an entry contributes at most ~12.5% of its raw Jaccard similarity.
    This prevents old generic entries (e.g. "Python async best practices")
    from diluting new specific sessions (e.g. "FastAPI rate limiting").
    """
    new_set = set(t.lower() for t in new_topic_keywords)
    if not new_set:
        return []

    relevant = []
    for entry in memory_entries:
        # Filter out low-quality entries — they're not useful context
        if entry.quality_score < 0.2:
            continue

        raw_sim = keyword_similarity(new_topic_keywords, entry.topic_keywords)
        recency = _recency_multiplier(entry.assimilated_at)
        adjusted_sim = raw_sim * recency

        if adjusted_sim >= threshold:
            relevant.append((adjusted_sim, recency, raw_sim, entry))

    # Sort by adjusted similarity descending, then by recency descending (fresher wins ties)
    relevant.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [entry for _, _, _, entry in relevant[:5]]


def inject_prior_context(
    prior_entries: list[MemoryEntry],
    max_findings_per_entry: int = 3,
) -> str:
    """
    Build a prior-research context block for injection into a new session prompt.

    Compresses MemoryEntries into a readable context block.
    Deduplicates by topic to avoid showing the same topic twice.
    """
    if not prior_entries:
        return ""

    # Deduplicate by topic (same topic = show only most recent)
    seen_topics: set[str] = set()
    unique_entries = []
    for entry in prior_entries:
        topic_clean = entry.topic.lower().strip()
        if topic_clean not in seen_topics:
            seen_topics.add(topic_clean)
            unique_entries.append(entry)

    lines = ["## Prior Research Context\n"]
    for entry in unique_entries:
        lines.append(f"**Topic:** {entry.topic}")
        lines.append(f"**Keywords:** {', '.join(entry.topic_keywords[:8])}")

        # Top findings
        if entry.key_findings:
            lines.append("**Key Findings:**")
            for finding in entry.key_findings[:max_findings_per_entry]:
                # Truncate long findings
                f = finding[:200] + "..." if len(finding) > 200 else finding
                lines.append(f"  • {f}")

        # Top source
        if entry.important_sources:
            top = entry.important_sources[0]
            lines.append(f"**Key Source:** [{top['title']}]({top['url']})")

        lines.append("")  # blank line between entries

    return '\n'.join(lines)


# ── Skill deduplication ──────────────────────────────────────

# Skill creation thresholds
SKILL_CREATION_THRESHOLD = 0.5  # Jaccard similarity to existing skill = skip
MIN_QUALITY_FOR_SKILL = 0.3     # quality score below this = skip
MIN_SOURCES_FOR_SKILL = 3       # minimum successful sources
MIN_KEYWORDS_FOR_SKILL = 3      # minimum keywords extracted


def should_create_skill(
    new_entry: MemoryEntry,
    existing_skills: list[MemoryEntry],
) -> tuple[bool, Optional[str]]:
    """
    Check if a MemoryEntry warrants a new research skill.

    Returns:
        (should_create, reason_or_existing_skill_name)
    """
    # Quality gate
    if new_entry.quality_score < MIN_QUALITY_FOR_SKILL:
        return False, f"quality too low ({new_entry.quality_score})"

    # Source gate
    successful_sources = len([
        s for s in new_entry.important_sources if s.get('type') != 'failed'
    ])
    if successful_sources < MIN_SOURCES_FOR_SKILL:
        return False, f"too few sources ({successful_sources})"

    # Keyword gate
    if len(new_entry.topic_keywords) < MIN_KEYWORDS_FOR_SKILL:
        return False, f"too few keywords ({len(new_entry.topic_keywords)})"

    # Similarity check against existing skills
    for skill in existing_skills:
        sim = topic_similarity(new_entry, skill)
        if sim >= SKILL_CREATION_THRESHOLD:
            return False, f"similar to existing skill: {skill.topic}"

    return True, None


# ── CLI ───────────────────────────────────────────────────────

if __name__ == '__main__':
    # Simple smoke test
    e1 = MemoryEntry(
        topic="Stripe webhook integration",
        topic_keywords=["stripe", "webhooks", "python", "hmac", "sha256", "idempotency"],
        key_findings=["Use HMAC-SHA256 to validate signatures", "Deduplicate by event ID"],
        important_sources=[{"title": "Stripe docs", "url": "https://stripe.com/docs/webhooks"}],
        code_patterns=["stripe.webhooks.construct_event()"],
        session_ids=["session_1"],
        assimilated_at="",
        depth=2,
        quality_score=0.8,
    )

    e2 = MemoryEntry(
        topic="Stripe payment processing",
        topic_keywords=["stripe", "payments", "api", "charge", "refund"],
        key_findings=[],
        important_sources=[],
        code_patterns=[],
        session_ids=["session_2"],
        assimilated_at="",
        depth=1,
        quality_score=0.5,
    )

    print(f"Jaccard: {keyword_similarity(e1.topic_keywords, e2.topic_keywords):.2f}")
    print(f"Topic sim: {topic_similarity(e1, e2):.2f}")
    print(f"Should create skill from e1 vs empty list: {should_create_skill(e1, [])}")
    print(f"Should create skill from e1 vs [e2]: {should_create_skill(e1, [e2])}")
