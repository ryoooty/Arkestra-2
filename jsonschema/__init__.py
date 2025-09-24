"""Minimal jsonschema subset for tests."""

from typing import Any, Dict


class ValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def validate(instance: Any, schema: Dict[str, Any]) -> None:
    """No-op validator placeholder."""

    return None


__all__ = ["validate", "ValidationError"]
