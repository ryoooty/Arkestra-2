"""Lightweight subset of Pydantic used in tests."""

from __future__ import annotations

from typing import Any, Dict, Set


def Field(default: Any = ..., **_kwargs: Any) -> Any:
    """Return provided default; metadata is ignored."""

    return default


class BaseModel:
    """Very small BaseModel implementation."""

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        self.__fields_set__: Set[str] = set()

        for name in annotations:
            if name in data:
                value = data[name]
                self.__fields_set__.add(name)
            else:
                value = getattr(self.__class__, name, ...)
            if value is ...:
                raise TypeError(f"Missing required field: {name}")
            setattr(self, name, value)

        for extra, value in data.items():
            if extra not in annotations:
                setattr(self, extra, value)
                self.__fields_set__.add(extra)

    def dict(self, *, exclude_unset: bool = False) -> Dict[str, Any]:
        annotations = getattr(self, "__annotations__", {})
        result: Dict[str, Any] = {}
        for name in annotations:
            if exclude_unset and name not in self.__fields_set__:
                continue
            result[name] = getattr(self, name)
        for name in getattr(self, "__dict__", {}):
            if name not in annotations:
                if exclude_unset and name not in self.__fields_set__:
                    continue
                result[name] = getattr(self, name)
        return result

    def model_dump(self, *, exclude_unset: bool = False) -> Dict[str, Any]:
        return self.dict(exclude_unset=exclude_unset)

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "BaseModel":
        return cls(**data)


__all__ = ["BaseModel", "Field"]
