"""Minimal FastAPI stub for tests without external dependency."""

from typing import Any, Callable, Dict, Tuple

RouteKey = Tuple[str, str]


class FastAPI:
    """Very small subset of FastAPI used in unit tests."""

    def __init__(self) -> None:
        self._routes: Dict[RouteKey, Callable[..., Any]] = {}

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a GET route."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[("GET", path)] = func
            return func

        return decorator

    def _dispatch(self, method: str, path: str, **kwargs: Any) -> Tuple[int, Any, Dict[str, Any]]:
        key = (method.upper(), path)
        handler = self._routes.get(key)
        if handler is None:
            raise KeyError(f"Route not found for {method} {path}")
        result = handler(**kwargs)
        status = 200
        headers: Dict[str, Any] = {}
        body = result
        if isinstance(result, tuple):
            if len(result) == 2:
                body, status = result
            elif len(result) >= 3:
                body, status, headers = result[0], result[1], result[2] or {}
        return status, body, headers


__all__ = ["FastAPI"]
