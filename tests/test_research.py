"""Tests for research package — unit tests for individual modules."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research.summarizer import (
    MemoryEntry,
    compute_quality_score,
    extract_keywords,
    extract_findings,
    extract_important_sources,
)
from research.topic_similarity import (
    keyword_similarity,
    topic_similarity,
    find_prior_research,
    inject_prior_context,
    should_create_skill,
    _recency_multiplier,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_session():
    return {
        "query": "FastAPI rate limiting with Redis",
        "turns": 2,
        "sources": [
            {
                "url": "https://fastapi.tiangolo.com/tutorial/rate-limiting/",
                "title": "FastAPI Rate Limiting",
                "fetch_status": "success",
                "content_length": 4500,
                "code_blocks": ["slowapi = SlowAPILimit(app)", "limiter.limit('5/minute')"],
                "summary": "Use slowapi for rate limiting in FastAPI. Specify limits as strings like '5/minute'.",
            },
            {
                "url": "https://github.com/laurentS/slowapi",
                "title": "slowapi",
                "fetch_status": "success",
                "content_length": 8200,
                "code_blocks": ["from slowapi import Limiter", "limiter = Limiter(key_func=get_remote_address)"],
                "summary": "SlowAPI is a rate limiting library for FastAPI built on top of limits.",
            },
            {
                "url": "https://redis.io/docs/manual/reactors/",
                "title": "Redis",
                "fetch_status": "success",
                "content_length": 3000,
                "code_blocks": ["SET rate:limit:user1 5", "EXPIRE rate:limit:user1 60"],
                "summary": "Use Redis for distributed rate limiting with INCR and EXPIRE.",
            },
            {
                "url": "https://example.com/blocked",
                "title": "Blocked",
                "fetch_status": "blocked",
                "content_length": 0,
                "code_blocks": [],
                "summary": "",
            },
        ],
        "findings": [
            "Use slowapi for FastAPI rate limiting with Redis store",
            "Rate limits configured as string literals like '100/minute'",
        ],
    }


@pytest.fixture
def sample_memory_entry():
    return MemoryEntry(
        topic="FastAPI rate limiting with Redis",
        topic_keywords=["fastapi", "rate", "limiting", "redis", "slowapi", "middleware"],
        key_findings=["Use slowapi for FastAPI rate limiting"],
        important_sources=[
            {"title": "slowapi", "url": "https://github.com/laurentS/slowapi"},
        ],
        code_patterns=["slowapi = SlowAPILimit(app)"],
        session_ids=["session_123"],
        assimilated_at="2025-05-01T12:00:00",
        depth=2,
        quality_score=0.8,
    )


# ── Keyword extraction ───────────────────────────────────────

class TestExtractKeywords:
    def test_tech_terms_extracted(self):
        text = "Use FastAPI with Python 3.11 and Redis for caching"
        kw = extract_keywords(text)
        assert "fastapi" in kw
        assert "python" in kw
        assert "redis" in kw

    def test_camel_case_extracted(self):
        text = "Use RequestHandler and CacheMiddleware for rate limiting"
        kw = extract_keywords(text)
        assert any("requesthandler" in k or "cachemiddleware" in k or "middleware" in k for k in kw)

    def test_stopwords_filtered(self):
        text = "the and for with this that from your best new"
        kw = extract_keywords(text)
        assert len(kw) == 0

    def test_max_keywords(self):
        text = " ".join(["fastapi"] * 30)
        kw = extract_keywords(text, max_keywords=20)
        assert len(kw) <= 20


# ── Quality scoring ─────────────────────────────────────────

class TestQualityScore:
    def test_passed_floor(self, sample_session):
        score = compute_quality_score(sample_session)
        assert score["passed_floor"] is True
        assert score["weighted"] > 0

    def test_signal_d_from_content_length(self, sample_session):
        score = compute_quality_score(sample_session)
        # 3/4 sources succeeded → 0.75
        assert score["signal_d"] == 0.75

    def test_failed_floor_too_few_sources(self):
        session = {
            "query": "test",
            "findings": ["finding"],
            "sources": [
                {"url": "https://a.com", "fetch_status": "success", "content_length": 100, "code_blocks": []},
                {"url": "https://b.com", "fetch_status": "blocked", "content_length": 0, "code_blocks": []},
            ],
        }
        score = compute_quality_score(session)
        assert score["passed_floor"] is False
        assert score["weighted"] == 0.0

    def test_failed_floor_no_code_blocks(self):
        session = {
            "query": "test",
            "findings": ["finding"],
            "sources": [
                {"url": "https://a.com", "fetch_status": "success", "content_length": 100, "code_blocks": []},
                {"url": "https://b.com", "fetch_status": "success", "content_length": 100, "code_blocks": []},
                {"url": "https://c.com", "fetch_status": "success", "content_length": 100, "code_blocks": []},
            ],
        }
        score = compute_quality_score(session)
        assert score["passed_floor"] is False


# ── Prior research injection ────────────────────────────────

class TestPriorResearch:
    def test_recency_multiplier_fresh(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(hours=1)).isoformat()
        m = _recency_multiplier(fresh)
        assert 0.98 < m <= 1.0

    def test_recency_multiplier_30_days(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=30)).isoformat()
        m = _recency_multiplier(old)
        assert 0.45 < m < 0.55  # ~0.5

    def test_recency_multiplier_90_days(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=90)).isoformat()
        m = _recency_multiplier(old)
        assert m == 0.125  # capped

    def test_find_prior_research_respects_recency(self, sample_memory_entry):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)

        old_entry = MemoryEntry(
            topic="Python async best practices",
            topic_keywords=["python", "async", "await", "asyncio"],
            key_findings=["Use asyncio.gather"],
            important_sources=[],
            code_patterns=[],
            session_ids=["s1"],
            assimilated_at=(now - timedelta(days=60)).isoformat(),
            depth=2,
            quality_score=0.7,
        )
        fresh_entry = MemoryEntry(
            topic="FastAPI rate limiting",
            topic_keywords=["fastapi", "rate", "limiting", "redis"],
            key_findings=["Use slowapi"],
            important_sources=[],
            code_patterns=[],
            session_ids=["s2"],
            assimilated_at=(now - timedelta(days=2)).isoformat(),
            depth=2,
            quality_score=0.8,
        )

        results = find_prior_research(
            ["fastapi", "rate", "limiting", "redis"],
            [old_entry, fresh_entry],
            threshold=0.1,
        )
        # Fresh entry should rank first (same or higher adjusted score)
        assert results[0].topic == "FastAPI rate limiting"

    def test_inject_prior_context_deduplicates_by_topic(self, sample_memory_entry):
        dup = MemoryEntry(
            topic="FastAPI rate limiting with Redis",  # exact same topic
            topic_keywords=["fastapi", "rate", "limiting"],
            key_findings=["Another finding"],
            important_sources=[],
            code_patterns=[],
            session_ids=["s3"],
            assimilated_at="2025-05-01T12:00:00",
            depth=1,
            quality_score=0.6,
        )
        block = inject_prior_context([sample_memory_entry, dup])
        assert block.count("**Topic:**") == 1  # deduped


# ── Skills ─────────────────────────────────────────────────

class TestShouldCreateSkill:
    def test_low_quality_rejected(self, sample_memory_entry):
        low_q = MemoryEntry(
            topic="test", topic_keywords=["a", "b", "c"],
            key_findings=["x"], important_sources=[{"url": "https://a.com"}],
            code_patterns=["x"], session_ids=["s1"],
            assimilated_at="", depth=1, quality_score=0.1,
        )
        create, reason = should_create_skill(low_q, [])
        assert create is False

    def test_similar_to_existing_skill_rejected(self, sample_memory_entry):
        existing = MemoryEntry(
            topic="FastAPI rate limiting",
            topic_keywords=["fastapi", "rate", "limiting", "redis"],
            key_findings=["x"], important_sources=[],
            code_patterns=[], session_ids=["s2"],
            assimilated_at="", depth=1, quality_score=0.8,
        )
        create, reason = should_create_skill(sample_memory_entry, [existing])
        assert create is False

    def test_distinct_topic_accepted(self, sample_memory_entry):
        distinct = MemoryEntry(
            topic="Stripe webhook integration",
            topic_keywords=["stripe", "webhook", "hmac"],
            key_findings=["x"],
            important_sources=[
                {"url": "https://stripe.com"},
                {"url": "https://stripe.com/docs"},
                {"url": "https://github.com/stripe"},
            ],
            code_patterns=["stripe.webhook"], session_ids=["s2"],
            assimilated_at="", depth=1, quality_score=0.85,
        )
        create, reason = should_create_skill(distinct, [])
        assert create is True


# ── Topic similarity ────────────────────────────────────────

class TestKeywordSimilarity:
    def test_jaccard_identical(self):
        sim = keyword_similarity(["a", "b", "c"], ["a", "b", "c"])
        assert sim == 1.0

    def test_jaccard_partial(self):
        sim = keyword_similarity(["a", "b", "c"], ["b", "c", "d"])
        assert 0.3 < sim < 0.6

    def test_jaccard_empty(self):
        sim = keyword_similarity([], [])
        assert sim == 0.0

    def test_jaccard_case_insensitive(self):
        sim = keyword_similarity(["FastAPI"], ["fastapi"])
        assert sim == 1.0
