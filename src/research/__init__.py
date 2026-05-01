"""
hermes-research: Autonomous multi-turn web research engine with memory and MCP server.
"""

from .engine import ResearchEngine, ResearchQuery, ResearchReport
from .assimilator import Assimilator
from .summarizer import (
    Summarizer,
    MemoryEntry,
    compute_quality_score,
    extract_keywords,
    extract_findings,
    extract_important_sources,
    extract_code_patterns,
)
from .topic_similarity import (
    find_prior_research,
    inject_prior_context,
    should_create_skill,
    keyword_similarity,
    topic_similarity,
)
from .skills_registry import SkillsRegistry
from .memory import Memory, ResearchSession, Source
from .search import WebSearch, ImageSearch, ArxivSearch
from .fetchers import WebFetcher, PaperFetcher

__all__ = [
    # Core engine
    "ResearchEngine",
    "ResearchQuery",
    "ResearchReport",
    # Assimilation loop
    "Assimilator",
    # Summarizer
    "Summarizer",
    "MemoryEntry",
    "compute_quality_score",
    "extract_keywords",
    "extract_findings",
    "extract_important_sources",
    "extract_code_patterns",
    # Similarity
    "find_prior_research",
    "inject_prior_context",
    "should_create_skill",
    "keyword_similarity",
    "topic_similarity",
    # Registry
    "SkillsRegistry",
    # Memory
    "Memory",
    "ResearchSession",
    "Source",
    # Search
    "WebSearch",
    "ImageSearch",
    "ArxivSearch",
    # Fetchers
    "WebFetcher",
    "PaperFetcher",
]
