from typing import Any, Dict


class Jinja2Templates:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def TemplateResponse(self, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"template": name, "context": context}


__all__ = ["Jinja2Templates"]
