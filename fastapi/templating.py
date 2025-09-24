"""Minimal Jinja2 templating stub for FastAPI."""

from typing import Any, Dict


class Jinja2Templates:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def TemplateResponse(self, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        response = {"template": name}
        response.update(context)
        return response


__all__ = ["Jinja2Templates"]
