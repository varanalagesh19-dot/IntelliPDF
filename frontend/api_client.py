"""Transport layer between the Streamlit UI and the IntelliPDF API.

Two modes, one interface:

* **HTTP** (default, local development) - the UI on :8501 talks to a separate
  FastAPI process on :8000 over ``requests``.
* **In-process** - the FastAPI app is mounted inside the Streamlit process via
  ``fastapi.testclient.TestClient`` and called directly.  This is what
  single-process hosts such as Streamlit Community Cloud need, where there is
  no room to run a second server.

Both expose ``request(method, path, ...)`` returning an object with
``status_code``, ``json()`` and ``text``, so ``frontend/app.py`` does not care
which one is active.

Selection order:

1. ``INTELLIPDF_INPROCESS=1`` forces in-process mode
2. ``API_URL`` reachable  -> HTTP mode
3. otherwise              -> in-process mode (single-process fallback)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600


class ResponseLike(Protocol):
    """Minimal response surface used by the UI."""

    status_code: int

    def json(self) -> Any: ...

    def text(self) -> str: ...


class HttpTransport:
    """Talk to a separately running FastAPI server."""

    mode = "http"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Any = None,
        files: Any = None,
        data: Any = None,
        timeout: int = DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ResponseLike:
        """Send an HTTP request to the API."""
        return requests.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            params=params,
            files=files,
            data=data,
            timeout=timeout,
        )

    def reachable(self, timeout: float = 15.0) -> bool:
        """Return ``True`` when the API answers its health probe."""
        for path in ("/health", "/api/health", ""):
            try:
                if self.request("GET", path, timeout=timeout).status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False


class InProcessTransport:
    """Call the FastAPI app inside this process (single-host deployments)."""

    mode = "in-process"

    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        from backend.main import app as fastapi_app
        from backend.utils import db as db_module
        from backend.config import ensure_directories

        ensure_directories()
        db_module.init_db()
        # Lifespan (model warm-up) is run explicitly: TestClient only triggers
        # it inside a context manager, and we want the app ready immediately.
        try:
            with TestClient(fastapi_app) as client:
                self._client = client
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
            LOGGER.warning("Could not start the API lifespan: %s", exc)
            self._client = TestClient(fastapi_app)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Any = None,
        files: Any = None,
        data: Any = None,
        timeout: int = DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ResponseLike:
        """Dispatch a request straight into the FastAPI app."""
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params is not None:
            kwargs["params"] = params
        if files is not None:
            kwargs["files"] = files
        if data is not None:
            kwargs["data"] = data
        return self._client.request(method, path, **kwargs)

    def reachable(self, timeout: float = 3.0) -> bool:
        """In-process transport is reachable by construction."""
        return True


_transport: HttpTransport | InProcessTransport | None = None


def get_transport(force: str | None = None) -> HttpTransport | InProcessTransport:
    """Return the shared transport, choosing a mode on first use.

    Args:
        force: ``"http"`` or ``"in-process"`` to override auto-detection.

    Returns:
        The process-wide transport instance.
    """
    global _transport
    if _transport is not None and force is None:
        return _transport
    if force == "http" and isinstance(_transport, HttpTransport):
        return _transport

    base_url = (os.getenv("API_BASE_URL") or os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")
    flag = (os.getenv("INTELLIPDF_INPROCESS") or "").strip().lower()

    if flag in {"1", "true", "yes"}:
        chosen: HttpTransport | InProcessTransport = InProcessTransport()
        LOGGER.info("IntelliPDF transport: in-process (forced)")
    else:
        http = HttpTransport(base_url)
        if http.reachable():
            chosen = http
            LOGGER.info("IntelliPDF transport: http -> %s", base_url)
        else:
            chosen = InProcessTransport()
            LOGGER.info(
                "IntelliPDF transport: in-process (no API reachable at %s)", base_url
            )

    _transport = chosen
    return chosen


def reset_transport() -> None:
    """Drop the cached transport (used by tests)."""
    global _transport
    _transport = None