"""Minimal responses stubs for FastAPI."""

from typing import Any, Dict


class HTMLResponse:
    def __init__(self, content: Any = None, status_code: int = 200, headers: Dict[str, Any] | None = None) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class RedirectResponse(HTMLResponse):
    def __init__(self, url: str, status_code: int = 307, headers: Dict[str, Any] | None = None) -> None:
        super().__init__(content=None, status_code=status_code, headers=headers)
        self.headers.setdefault("Location", url)


__all__ = ["HTMLResponse", "RedirectResponse"]
