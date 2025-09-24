from typing import Any

from app.util import simple_yaml


def safe_load(data: str) -> Any:
    return simple_yaml.loads(data)


__all__ = ["safe_load"]
