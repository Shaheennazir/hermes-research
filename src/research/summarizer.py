#!/usr/bin/env python3
"""
Summarizer — compress a raw research session into a memory-ready format.

Extracts:
- topic keywords (for similarity matching)
- key findings (top insights, deduplicated)
- important sources (canonical URLs, best sources)
- code patterns (distinct snippets)
"""

import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


# ── Data structures ────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """What gets stored in Hermes memory after research."""
    topic: str                    # original query, cleaned
    topic_keywords: list[str]    # Jaccard-matching keywords
    key_findings: list[str]       # max 10, deduplicated
    important_sources: list[dict] # canonical sources with URLs and titles
    code_patterns: list[str]      # distinct code snippets
    session_ids: list[str]        # which sessions contributed
    assimilated_at: str           # ISO timestamp
    depth: int                    # max turns across sessions
    quality_score: float          # weighted quality score (0.0-1.0)
    quality_breakdown: dict = field(default_factory=dict)  # Signal A/B/C/D breakdown

    def to_dict(self) -> dict:
        return asdict(self)


# ── Keyword extraction ────────────────────────────────────────

# Technologies, frameworks, packages, concepts we want to tag
TECH_TERMS = re.compile(
    r'\b('
    # JS/TS ecosystems
    r'next\.js|react|typescript|node\.js|express|fastify|'
    r'tailwind|postgresql|mongodb|redis|graphql|rest\s?api|'
    # Python
    r'python|django|flask|fastapi|pydantic|sqlalchemy|'
    # Cloud
    r'aws|azure|gcp|kubernetes|docker|terraform|'
    # AI/ML
    r'openai|anthropic|llm|gpt|claude|rag|vector\s?db|'
    # Infra
    r'vercel|netlify|cloudflare|nginx|postgres|mysql|'
    # Auth
    r'oauth|jwt|ldap|saml|webhook|'
    # Patterns
    r'crud|cache|queue|websocket|microservice|monolith|'
    # General web
    r'http|https|tcp|grpc|webhook|cdn|ssr|csr|ssg|'
    # Databases
    r'sql|nosql|orm|acidity|transaction|indexing|'
    # Concurrency
    r'async|await|threading|multiprocessing|worker|'
    # Security
    r'xss|csrf|sql\s?injection|oauth|sso|2fa|mfa|'
    # Misc
    r'cli|api\s?key|rate\s?limit|pagination|cors|json|xml|yaml|jsonld'
    r')\b',
    re.I
)

# Common stopwords to filter from keywords
STOPWORDS = {
    'the', 'and', 'for', 'with', 'this', 'that', 'from', 'your', 'their',
    'what', 'when', 'where', 'how', 'why', 'which', 'who', 'will', 'can',
    'use', 'using', 'used', 'also', 'are', 'was', 'were', 'been', 'have',
    'has', 'had', 'not', 'but', 'into', 'all', 'any', 'each', 'more',
    'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'about', 'over', 'under', 'after', 'before', 'between', 'through',
    'during', 'above', 'below', 'should', 'could', 'would', 'might',
    'must', 'shall', 'does', 'did', 'doing', 'being', 'here', 'there',
    'because', 'while', 'although', 'unless', 'until', 'both', 'either',
    'neither', 'whether', 'if', 'then', 'else', 'when', 'where', 'which',
    'server', 'client', 'data', 'app', 'apps', 'web', 'page', 'pages',
    'best', 'new', 'learn', 'build', 'using', 'guide', 'tutorial',
    'example', 'implementation', 'how', 'to', 'way', 'method', 'approach',
    'article', 'post', 'blog', 'docs', 'documentation', 'com', 'org', 'io',
    'https', 'http', 'www', 'github', 'stackoverflow', 'medium', 'dev',
}


def extract_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """Extract technical keywords from text using regex + filtering."""
    # Find all tech terms
    tech_matches = TECH_TERMS.findall(text)
    # Find capitalized CamelCase terms (class names, components)
    camel_matches = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    # Find dotted terms (package.module, domain.tld)
    dotted = re.findall(r'\b[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+\b', text, re.I)

    all_terms = []
    for term in tech_matches + camel_matches + dotted:
        t = term.lower().strip()
        if t not in STOPWORDS and len(t) > 2:
            all_terms.append(t)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in all_terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:max_keywords]


# ── Finding extraction ────────────────────────────────────────

def extract_findings(sources: list[dict], findings: list[str], max_findings: int = 10) -> list[str]:
    """
    Extract key findings from research session.

    Findings from the engine (e.g. '[Turn 1] Key technologies found: next.js...')
    are already synthesized — include them directly.
    Then extract any specific facts from source content.
    """
    if not sources and not findings:
        return []

    findings_out = []

    # Engine-generated findings are already synthesized — include directly
    for f in findings:
        cleaned = f.strip()
        if len(cleaned) > 10:
            # Truncate if too long
            if len(cleaned) > 300:
                cleaned = cleaned[:300] + '...'
            findings_out.append(cleaned)

    # Collect all source content
    summaries = []
    for s in sources:
        fetch_status = s.get('fetch_status') or s.get('status', '')
        if fetch_status == 'success':
            if s.get('summary'):
                summaries.append(s['summary'])
            if s.get('snippet'):
                summaries.append(s['snippet'])

    combined_text = ' '.join(summaries)

    # Extract specific technical facts from source content:
    # - Version numbers
    # - Function/class signatures in code
    # - Configuration values
    important_patterns = [
        (r'[^.!?]*\b(?:must|requires|uses|supports|provides|returns|accepts)\b[^.!?]{5,150}[.!?]', 0, 150),
        (r'[^.!?]*\bv?\d+(?:\.\d+)+(?:-\w+)?\b[^.!?]{5,150}[.!?]', 0, 150),
        (r'[^.!?]*`(?:[^`]{10,100})`[^.!?]*[.!?]', 0, 0),
    ]

    for pattern, min_len, max_len in important_patterns:
        matches = re.findall(pattern, combined_text, re.I)
        for m in matches:
            cleaned = m.strip()
            # Skip if too short or too long
            if len(cleaned) < 20:
                continue
            if max_len and len(cleaned) > max_len:
                cleaned = cleaned[:max_len] + '...'
            # Deduplicate
            if cleaned not in findings_out:
                findings_out.append(cleaned)

    # Deduplicate findings that are too similar
    deduped = []
    for f in findings_out:
        is_duplicate = False
        words_f = set(f.lower().split())
        for existing in deduped:
            words_e = set(existing.lower().split())
            if len(words_f) > 0 and len(words_e) > 0:
                overlap = len(words_f & words_e) / max(len(words_f), len(words_e))
                if overlap > 0.7:
                    is_duplicate = True
                    break
        if not is_duplicate:
            deduped.append(f)

    return deduped[:max_findings]


# ── Source extraction ─────────────────────────────────────────

def extract_important_sources(sources: list[dict], max_sources: int = 8) -> list[dict]:
    """Pick the most important sources — prefer docs, official sources, unique domains."""
    if not sources:
        return []

    def source_score(s: dict) -> tuple[float, str]:
        score = 0.0
        url = s.get('url', '')
        url_lower = url.lower()

        # Prefer success status — check both field names
        fetch_status = s.get('fetch_status') or s.get('status', '')
        if fetch_status == 'success':
            score += 1.0

        # Prefer docs.official domains
        if any(d in url_lower for d in ['docs.', 'documentation.', 'developer.', 'api.']):
            score += 2.0

        # Penalize social media and aggregators
        if any(d in url_lower for d in ['twitter.com', 'x.com', 'facebook.com', 'reddit.com']):
            score -= 1.0

        # Prefer sources with code — field is code_blocks in actual sessions
        code_blocks = s.get('code_blocks') or s.get('code', [])
        if code_blocks:
            score += 0.5

        # Prefer longer summaries (more content)
        summary = s.get('summary', s.get('snippet', ''))
        score += len(summary) / 10000.0

        # Prefer unique root paths (allow multiple pages from same domain, different sections)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # Use domain + first two path segments as the dedup key
            path_parts = [p for p in parsed.path.split('/') if p][:2]
            path_key = f"{parsed.netloc}/{'/'.join(path_parts)}"
            score += len(path_key) / 100  # slight preference for real paths
        except Exception:
            path_key = url

        return score, path_key

    # Filter to successful sources
    valid_sources = [s for s in sources if (s.get('fetch_status') or s.get('status')) == 'success']
    if not valid_sources:
        return []

    # Sort by score
    scored = [(source_score(s)[0], source_score(s)[1], s) for s in valid_sources]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Pick top sources, deduplicating by root path
    chosen = []
    paths_seen = set()
    for score, path_key, s in scored:
        if path_key not in paths_seen:
            chosen.append({
                'title': s.get('title', '')[:100],
                'url': s['url'],
                'type': s.get('source_type', s.get('type', 'web')),
            })
            paths_seen.add(path_key)
            if len(chosen) >= max_sources:
                break

    return chosen


# ── Code pattern extraction ────────────────────────────────────

def extract_code_patterns(sources: list[dict], max_patterns: int = 5) -> list[str]:
    """Extract distinct code patterns from sources."""
    all_snippets = []
    for s in sources:
        # Support both field names
        code = s.get('code_blocks') or s.get('code', [])
        if code and isinstance(code, list):
            all_snippets.extend(code)

    if not all_snippets:
        return []

    # Deduplicate similar snippets
    patterns = []
    for snippet in all_snippets:
        # Normalize: lowercase, remove extra whitespace
        normalized = ' '.join(snippet.lower().split())
        is_duplicate = False
        for existing in patterns:
            # If 80% of the shorter one is in the longer one, call it dup
            shorter = normalized if len(normalized) < len(existing) else existing
            longer = existing if len(normalized) < len(existing) else normalized
            if len(shorter) > 20 and shorter in longer:
                is_duplicate = True
                break
        if not is_duplicate and len(normalized) > 20:
            patterns.append(normalized)

    return patterns[:max_patterns]


# ── Quality scoring ────────────────────────────────────────────

# Specificity anchors — binary per-finding check
VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b')
SIGNATURE_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.\w+)*\s*\([^)]{3,}\)')
HTTP_CODE_RE = re.compile(r'\b[1-5]\d{2}\b')  # 200, 404, 500, etc.
PORT_RE = re.compile(r'\b(?:port|tcp|udp):\s*\d+\b', re.I)
TIMEOUT_RE = re.compile(r'\b(?:timeout|delay|latency|threshold)\s*[=:]\s*\d+\b', re.I)
CONSTANT_RE = re.compile(r'\b(?:MAX_|MIN_|DEFAULT_|ERROR_|SUCCESS_|FAIL_)\w+\b')


def _has_specificity(finding: str) -> bool:
    """Binary check: does this finding contain a concrete technical anchor?"""
    return bool(
        VERSION_RE.search(finding)
        or SIGNATURE_RE.search(finding)
        or HTTP_CODE_RE.search(finding)
        or PORT_RE.search(finding)
        or TIMEOUT_RE.search(finding)
        or CONSTANT_RE.search(finding)
    )


def _source_diversity_score(sources: list[dict]) -> float:
    """Signal A: Jaccard similarity of domains. 1 unique = 0, 4+ unique = 1.0."""
    if not sources:
        return 0.0
    successful = [s for s in sources if (s.get('fetch_status') or s.get('status')) == 'success']
    if not successful:
        return 0.0

    def get_domain(s: dict) -> str:
        url = s.get('url', '')
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return url.split('/')[2] if '/' in url else url

    domains = set(get_domain(s) for s in successful)
    unique = len(domains)
    return min(unique / 4.0, 1.0)


def _finding_specificity_score(findings: list[str]) -> float:
    """Signal B: % of findings with at least one concrete technical anchor."""
    if not findings:
        return 0.0
    hits = sum(1 for f in findings if _has_specificity(f))
    return hits / len(findings)


def _code_relevance_score(sources: list[dict], query: str) -> float:
    """
    Signal C: code snippets are long enough + contain query-relevant keywords.

    Two gates:
    1. snippet length > 50 chars (separates real code from boilerplate)
    2. snippet contains at least one keyword from the query
    """
    # Extract query keywords
    query_keywords = set(
        w.lower() for w in re.findall(r'\b\w{4,}\b', query)
        if w.lower() not in {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'what',
            'when', 'where', 'how', 'why', 'best', 'using', 'guide',
            'tutorial', 'example', 'implementation', 'about', 'learn',
        }
    )

    successful = [s for s in sources if (s.get('fetch_status') or s.get('status')) == 'success']
    all_snippets = []
    for s in successful:
        code = s.get('code_blocks') or s.get('code', [])
        if code and isinstance(code, list):
            all_snippets.extend(code)

    if not all_snippets:
        return 0.0

    relevant_count = 0
    for snippet in all_snippets:
        # Gate 1: length
        if len(snippet) < 50:
            continue
        # Gate 2: keyword overlap with query
        snippet_lower = snippet.lower()
        if query_keywords and any(kw in snippet_lower for kw in query_keywords):
            relevant_count += 1
        elif not query_keywords:
            # No query keywords — just check it's not pure boilerplate
            # Assume > 100 chars with non-alphanumeric chars = real code
            if len(snippet) > 100 and re.search(r'[{}=;()"]+', snippet):
                relevant_count += 1

    return min(relevant_count / max(len(all_snippets), 1), 1.0)


def _fetch_success_rate(sources: list[dict]) -> float:
    """
    Signal D: % of fetched sources that returned actual content.
    Calculated at fetcher level, stored in content_length.
    """
    if not sources:
        return 0.0

    total = len(sources)
    successful = sum(
        1 for s in sources
        if (s.get('fetch_status') or s.get('status')) == 'success'
        and (s.get('content_length', 0) > 0 or s.get('code_blocks') or s.get('code'))
    )

    return successful / total


def compute_quality_score(session_data: dict) -> dict:
    """
    Compute quality score from signals A/B/C/D.

    Returns a dict with all components for transparency:
    {
      'signal_a': float,  # source diversity (0-1)
      'signal_b': float,  # finding specificity (0-1)
      'signal_c': float,  # code relevance (0-1)
      'signal_d': float,  # fetch success rate (0-1)
      'weighted': float,  # final weighted score
      'passed_floor': bool,
    }

    Floor: ≥ 3 successful sources AND ≥ 2 code blocks required.
    Below floor: score is 0 regardless of signals.
    """
    sources = session_data.get('sources', [])
    findings = session_data.get('findings', [])
    query = session_data.get('query', '')

    # Floor check
    successful_sources = [
        s for s in sources
        if (s.get('fetch_status') or s.get('status')) == 'success'
    ]
    total_code_blocks = sum(
        len(s.get('code_blocks') or s.get('code', []))
        for s in successful_sources
    )

    passed_floor = len(successful_sources) >= 3 and total_code_blocks >= 2

    if not passed_floor:
        return {
            'signal_a': 0.0, 'signal_b': 0.0,
            'signal_c': 0.0, 'signal_d': 0.0,
            'weighted': 0.0,
            'passed_floor': False,
        }

    sig_a = _source_diversity_score(sources)
    sig_b = _finding_specificity_score(findings)
    sig_c = _code_relevance_score(sources, query)
    sig_d = _fetch_success_rate(sources)

    weighted = round(0.20 * sig_a + 0.15 * sig_b + 0.30 * sig_c + 0.35 * sig_d, 3)

    return {
        'signal_a': round(sig_a, 3),
        'signal_b': round(sig_b, 3),
        'signal_c': round(sig_c, 3),
        'signal_d': round(sig_d, 3),
        'weighted': weighted,
        'passed_floor': True,
    }


# ── Main summarizer ────────────────────────────────────────────

class Summarizer:
    """Convert raw session data into a MemoryEntry."""

    def summarize(self, session_id: str, session_path: Optional[Path] = None) -> MemoryEntry:
        """
        Load session file and produce a MemoryEntry.

        Args:
            session_id: the session ID (e.g. 'session_1234567890')
            session_path: optional explicit path; otherwise uses default dir

        Returns:
            MemoryEntry ready for Hermes memory
        """
        if session_path is None:
            session_path = Path.home() / '.hermes' / 'research_sessions' / f'{session_id}.json'

        data = json.loads(session_path.read_text())

        keywords = extract_keywords(
            data.get('query', '') + ' ' + ' '.join(data.get('findings', []))
        )

        findings = extract_findings(
            data.get('sources', []),
            data.get('findings', [])
        )

        important_sources = extract_important_sources(data.get('sources', []))
        code_patterns = extract_code_patterns(data.get('sources', []))

        quality_result = compute_quality_score(data)

        return MemoryEntry(
            topic=data.get('query', 'Unknown'),
            topic_keywords=keywords,
            key_findings=findings,
            important_sources=important_sources,
            code_patterns=code_patterns,
            session_ids=[session_id],
            assimilated_at='',  # filled by caller
            depth=data.get('research_depth', 1),
            quality_score=quality_result['weighted'],
            quality_breakdown=quality_result,
        )


# ── CLI ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    s = Summarizer()
    entry = s.summarize(session_id)
    print(json.dumps(entry.to_dict(), indent=2))
