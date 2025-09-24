from __future__ import annotations

from typing import Any, Dict


class BaseModel:
    __fields__: set[str]

    def __init_subclass__(cls) -> None:
        cls.__fields__ = set(getattr(cls, "__annotations__", {}).keys())

    def __init__(self, **data: Any) -> None:
        provided = set(data.keys())
        object.__setattr__(self, "__provided_fields", provided)
        for name in getattr(self, "__fields__", set()):
            if name in data:
                value = data[name]
            else:
                value = getattr(self.__class__, name, None)
            setattr(self, name, value)

    def dict(self, *, exclude_unset: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        provided = getattr(self, "__provided_fields", set())
        for name in getattr(self, "__fields__", set()):
            if exclude_unset and name not in provided:
                continue
            result[name] = getattr(self, name, None)
        return result


def Field(default: Any = ..., **_: Any) -> Any:
    return default


__all__ = ["BaseModel", "Field"]
