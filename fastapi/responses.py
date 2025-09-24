from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HTMLResponse:
    content: Any = None
    status_code: int = 200
    headers: Optional[Dict[str, Any]] = None


@dataclass
class RedirectResponse:
    url: str
    status_code: int = 307
    headers: Optional[Dict[str, Any]] = None


__all__ = ["HTMLResponse", "RedirectResponse"]
