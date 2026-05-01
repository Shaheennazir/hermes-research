#!/usr/bin/env python3
"""
Content fetchers — web pages, papers.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


@dataclass
class FetchResult:
    url: str
    title: str
    content: str  # main text content
    code_blocks: list[str]
    fetch_status: str  # success | blocked | failed
    source_type: str = "web"
    content_length: int = 0  # raw char count of content (Signal D input)


class ContentFetcher(ABC):
    @abstractmethod
    def fetch(self, url: str) -> FetchResult:
        ...


class WebFetcher(ContentFetcher):
    """Fetch web pages and extract content + code."""

    BLOCKED_HOSTS = [
        "medium.com", "twitter.com", "x.com",
        "facebook.com", "linkedin.com",
        "web.archive.org",
    ]

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    ]

    def fetch(self, url: str) -> FetchResult:
        if not requests or not BeautifulSoup:
            return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)

        if any(b in url.lower() for b in self.BLOCKED_HOSTS):
            return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="blocked", content_length=0)

        for attempt, ua in enumerate(self.USER_AGENTS):
            try:
                headers = {
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                }
                resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)

                soup = BeautifulSoup(resp.text, "lxml")

                # Remove noise
                for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "iframe"]):
                    tag.decompose()
                for noise in soup.find_all(class_=re.compile(r"ad-|paywall|newsletter|cookie|popup|modal|sidebar", re.I)):
                    noise.decompose()

                # Title
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""

                # Content selectors (best to worst)
                for selector in [
                    "article", "main", '[role="main"]',
                    ".post-content", ".entry-content", ".article-content",
                    ".content", "#content", ".post", ".article",
                ]:
                    main = soup.select_one(selector)
                    if main and len(main.get_text(strip=True)) > 200:
                        content = main.get_text(separator="\n", strip=True)
                        break
                else:
                    body = soup.find("body")
                    content = body.get_text(separator="\n", strip=True) if body else ""

                code_blocks = self._extract_code(content)
                raw_length = len(content)

                return FetchResult(
                    url=url,
                    title=title,
                    content=self._clean(content[:5000]),
                    code_blocks=code_blocks,
                    fetch_status="success",
                    content_length=raw_length,
                )

            except Exception:
                if attempt == len(self.USER_AGENTS) - 1:
                    return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)
                time.sleep(0.5)
                continue

        return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)

    def _extract_code(self, text: str) -> list[str]:
        lines = text.split("\n")
        blocks = []
        current = []

        code_indicators = [
            "import ", "from ", "export ", "const ", "let ", "var ",
            "function ", "async ", "await ", "return ", "class ",
            "def ", "fn ", "pub ", "struct ", "interface ",
            "npm ", "yarn ", "pip ", "cargo ", "```",
            "interface ", "type ", "enum ", "namespace ",
        ]

        for line in lines:
            stripped = line.strip()
            is_code = (
                stripped.startswith("```")
                or any(stripped.startswith(ind) for ind in code_indicators)
                or re.match(r"^\s*(if|for|while|switch|try|catch).*\(.*\)\s*{", stripped)
                or re.match(r"^\s*//", stripped)
                or re.match(r"^\s*#\s*\w+", stripped)
            )
            if is_code:
                current.append(line)
            elif current and stripped == "":
                if len("\n".join(current).strip()) > 20:
                    blocks.append("\n".join(current))
                current = []
            elif current:
                current.append(line)

        if current and len("\n".join(current).strip()) > 20:
            blocks.append("\n".join(current))

        return blocks[:5]

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


class PaperFetcher(ContentFetcher):
    """Fetch arXiv paper abstract pages."""

    def fetch(self, url: str) -> FetchResult:
        if not requests or not BeautifulSoup:
            return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)

        if "arxiv.org" not in url:
            return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", content_length=0)

        arxiv_url = url if "/abs/" in url else url.replace("/pdf/", "/abs/")
        if not arxiv_url.endswith("/abs/") and not arxiv_url.endswith("/abs"):
            arxiv_url = arxiv_url.rstrip("/") + "/abs"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +http://example.com/bot)",
                "Accept": "text/html",
            }
            resp = requests.get(arxiv_url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            title_tag = soup.find("meta", attrs={"name": "citation_title"})
            title = title_tag["content"] if title_tag else ""

            abstract_tag = soup.find("meta", attrs={"name": "description"})
            abstract = abstract_tag["content"].strip() if abstract_tag else ""

            author_tags = soup.find_all("meta", attrs={"name": "citation_author"})
            authors = [a["content"] for a in author_tags]

            content = f"Title: {title}\nAuthors: {', '.join(authors)}\n\nAbstract: {abstract}"

            return FetchResult(
                url=url,
                title=title,
                content=content,
                code_blocks=[],
                fetch_status="success",
                source_type="paper",
                content_length=len(content),
            )
        except Exception:
            return FetchResult(url=url, title="", content="", code_blocks=[], fetch_status="failed", source_type="paper", content_length=0)
