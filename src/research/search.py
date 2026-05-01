#!/usr/bin/env python3
"""
Search backends — web (mmx), image (mmx), arXiv paper search.
"""

import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    date: str = ""
    source: str = "web"


class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, num_results: int = 8) -> list[SearchResult]:
        ...


class WebSearch(SearchBackend):
    """mmx search — web search"""

    def __init__(self):
        self.cache: dict[str, list[SearchResult]] = {}

    def search(self, query: str, num_results: int = 8) -> list[SearchResult]:
        if query in self.cache:
            return self.cache[query][:num_results]

        result = subprocess.run(
            ["mmx", "search", "query", query],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return []

        try:
            data = json.loads(result.stdout)
            organic = data.get("organic", [])
            results = [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    date=item.get("date", ""),
                    source="web",
                )
                for item in organic
                if item.get("link")
            ]
            self.cache[query] = results
            return results[:num_results]
        except (json.JSONDecodeError, KeyError):
            return []


class ImageSearch(SearchBackend):
    """mmx vision — image search via web search for image URLs"""

    def __init__(self):
        self.cache: dict[str, list[SearchResult]] = {}

    def search(self, query: str, num_results: int = 8) -> list[SearchResult]:
        # Use mmx search with image-focused query modifiers
        image_query = f"{query} site:pinterest.com OR site:unsplash.com OR site:pexels.com OR site:images.unsplash.com"
        result = subprocess.run(
            ["mmx", "search", "query", image_query],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return []

        try:
            data = json.loads(result.stdout)
            organic = data.get("organic", [])
            results = [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    date=item.get("date", ""),
                    source="image",
                )
                for item in organic[:num_results]
                if item.get("link")
            ]
            self.cache[query] = results
            return results
        except (json.JSONDecodeError, KeyError):
            return []


class ArxivSearch(SearchBackend):
    """arXiv paper search via their Atom API"""

    ARXIV_API = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.cache: dict[str, list[SearchResult]] = {}

    def search(self, query: str, num_results: int = 8) -> list[SearchResult]:
        if query in self.cache:
            return self.cache[query][:num_results]

        import urllib.parse
        import urllib.request

        params = {
            "search_query": f"all:{urllib.parse.quote(query)}",
            "start": 0,
            "max_results": num_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.ARXIV_API}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ResearchBot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                import xml.etree.ElementTree as ET
                tree = ET.parse(resp)
                root = tree.getroot()
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

                results = []
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    link = entry.find("atom:id", ns)
                    published = entry.find("atom:published", ns)

                    title_text = title.text.strip().replace("\n", " ") if title is not None else ""
                    snippet_text = summary.text.strip().replace("\n", " ")[:300] if summary is not None else ""
                    url_text = link.text.strip() if link is not None else ""
                    date_text = published.text[:10] if published is not None else ""

                    if title_text and url_text:
                        results.append(SearchResult(
                            title=title_text,
                            url=url_text,
                            snippet=snippet_text + "...",
                            date=date_text,
                            source="paper",
                        ))

                self.cache[query] = results
                return results[:num_results]
        except Exception:
            return []
