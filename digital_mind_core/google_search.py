"""Google public-search adapter for UnifiedKernel.

Two explicit transports are supported:

1. Native Google Programmable Search JSON API using environment variables:
   GOOGLE_API_KEY and GOOGLE_CSE_ID.
2. A host callback that is explicitly declared to be a Google-backed search.

No HTML scraping fallback is provided.  Missing Google credentials therefore
fails closed instead of silently switching to another search engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable
from urllib.parse import urlencode, urlparse
import urllib.request


API_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
MAX_QUERY_CHARS = 512
MAX_RESPONSE_BYTES = 2_000_000


class GoogleSearchUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GoogleSearchConfig:
    api_key: str | None = None
    search_engine_id: str | None = None
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls):
        return cls(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            search_engine_id=os.environ.get("GOOGLE_CSE_ID"),
        )

    def native_ready(self):
        return bool(self.api_key and self.search_engine_id)


class GoogleSearchClient:
    def __init__(
        self,
        config: GoogleSearchConfig | None = None,
        *,
        host_search: Callable[[str, int], dict[str, Any]] | None = None,
        host_declares_google: bool = False,
    ):
        self.config = config or GoogleSearchConfig.from_environment()
        self.host_search = host_search
        self.host_declares_google = bool(host_declares_google)
        self.search_count = 0
        self.last_transport = None

    def status(self):
        if self.config.native_ready():
            status = "READY_NATIVE_GOOGLE_API"
        elif self.host_search is not None and self.host_declares_google:
            status = "READY_HOST_GOOGLE"
        elif self.host_search is not None:
            status = "HOST_SEARCH_PRESENT_BUT_NOT_GOOGLE_VERIFIED"
        else:
            status = "NEEDS_GOOGLE_CREDENTIALS_OR_VERIFIED_HOST"
        return {
            "status": status,
            "native_api": self.config.native_ready(),
            "host_callback": self.host_search is not None,
            "host_declares_google": self.host_declares_google,
            "credentials_exposed": False,
            "search_count": self.search_count,
            "last_transport": self.last_transport,
        }

    @staticmethod
    def _validate_query(query, num):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
        if type(num) is not int or not 1 <= num <= 10:
            raise ValueError("num must be an integer from 1 to 10")
        return query.strip()

    @staticmethod
    def _clean_result(item):
        if not isinstance(item, dict):
            raise ValueError("Google result item must be an object")
        link = item.get("link")
        if not isinstance(link, str):
            raise ValueError("Google result has no link")
        parsed = urlparse(link)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Google result link is not a valid web URL")
        return {
            "title": item.get("title") if isinstance(item.get("title"), str) else "",
            "link": link,
            "snippet": item.get("snippet") if isinstance(item.get("snippet"), str) else "",
            "display_link": (
                item.get("displayLink")
                if isinstance(item.get("displayLink"), str)
                else parsed.netloc
            ),
        }

    def _native_search(self, query, num):
        params = {
            "key": self.config.api_key,
            "cx": self.config.search_engine_id,
            "q": query,
            "num": num,
        }
        url = API_ENDPOINT + "?" + urlencode(params)
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "digital-mind-unified-kernel-google-search/1",
            },
        )
        with urllib.request.urlopen(
            request, timeout=float(self.config.timeout_seconds)
        ) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise GoogleSearchUnavailable("Google response exceeds size limit")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise GoogleSearchUnavailable("Google returned invalid JSON")
        if "error" in payload:
            error = payload["error"]
            message = (
                error.get("message")
                if isinstance(error, dict) and isinstance(error.get("message"), str)
                else "Google Search API error"
            )
            raise GoogleSearchUnavailable(message)
        items = payload.get("items") or []
        if not isinstance(items, list):
            raise GoogleSearchUnavailable("Google results payload is malformed")
        return {
            "provider": "google_programmable_search",
            "query": query,
            "results": [self._clean_result(item) for item in items[:num]],
        }

    def _host_google_search(self, query, num):
        if self.host_search is None or not self.host_declares_google:
            raise GoogleSearchUnavailable(
                "host callback is not explicitly verified as Google-backed"
            )
        payload = self.host_search(query, num)
        if not isinstance(payload, dict):
            raise GoogleSearchUnavailable("host Google search must return an object")
        provider = payload.get("provider")
        if provider != "google":
            raise GoogleSearchUnavailable(
                "host callback did not attest provider='google'"
            )
        items = payload.get("results") or []
        if not isinstance(items, list):
            raise GoogleSearchUnavailable("host Google results are malformed")
        return {
            "provider": "google_host",
            "query": query,
            "results": [self._clean_result(item) for item in items[:num]],
        }

    def search(self, query, num=5):
        query = self._validate_query(query, num)
        if self.config.native_ready():
            result = self._native_search(query, num)
            self.last_transport = "native_google_api"
        elif self.host_search is not None and self.host_declares_google:
            result = self._host_google_search(query, num)
            self.last_transport = "host_google_callback"
        else:
            raise GoogleSearchUnavailable(
                "Google Search is not connected: set GOOGLE_API_KEY and "
                "GOOGLE_CSE_ID, or attach a host callback explicitly verified "
                "as Google-backed."
            )
        self.search_count += 1
        return result
