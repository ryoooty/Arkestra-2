"""Minimal FastAPI stub for tests without external dependency."""

from typing import Any, Callable, Dict, Tuple
import inspect

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency during tests
    BaseModel = None  # type: ignore[misc,assignment]

RouteKey = Tuple[str, str]


class FastAPI:
    """Very small subset of FastAPI used in unit tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._routes: Dict[RouteKey, tuple[Callable[..., Any], int | None]] = {}
        self._events: Dict[str, list[Callable[..., Any]]] = {}
        self._middleware: Dict[str, list[Callable[..., Any]]] = {}

    def _register(
        self,
        method: str,
        path: str,
        func: Callable[..., Any],
        *,
        status_code: int | None = None,
    ) -> Callable[..., Any]:
        self._routes[(method.upper(), path)] = (func, status_code)
        return func

    def get(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a GET route."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            status_code = kwargs.pop("status_code", None)
            return self._register("GET", path, func, status_code=status_code)

        return decorator

    def post(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a POST route."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            status_code = kwargs.pop("status_code", None)
            return self._register("POST", path, func, status_code=status_code)

        return decorator

    def put(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            status_code = kwargs.pop("status_code", None)
            return self._register("PUT", path, func, status_code=status_code)

        return decorator

    def delete(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            status_code = kwargs.pop("status_code", None)
            return self._register("DELETE", path, func, status_code=status_code)

        return decorator

    def on_event(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._events.setdefault(event, []).append(func)
            return func

        return decorator

    def middleware(self, mtype: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._middleware.setdefault(mtype, []).append(func)
            return func

        return decorator

    def _dispatch(self, method: str, path: str, **kwargs: Any) -> Tuple[int, Any, Dict[str, Any]]:
        key = (method.upper(), path)
        entry = self._routes.get(key)
        if entry is None:
            raise KeyError(f"Route not found for {method} {path}")
        handler, default_status = entry
        json_payload = kwargs.pop("json", None)
        if json_payload is not None:
            sig = inspect.signature(handler)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and p.name not in kwargs
            ]
            if len(params) == 1:
                param = params[0]
                value = json_payload
                if (
                    BaseModel is not None
                    and isinstance(param.annotation, type)
                    and issubclass(param.annotation, BaseModel)
                    and isinstance(json_payload, dict)
                ):
                    value = param.annotation(**json_payload)
                kwargs[param.name] = value
            elif isinstance(json_payload, dict):
                for name, param in sig.parameters.items():
                    if (
                        name not in kwargs
                        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                        and name in json_payload
                    ):
                        kwargs[name] = json_payload[name]
        try:
            result = handler(**kwargs)
        except HTTPException as exc:  # pragma: no cover - simple exception handling
            return exc.status_code, {"detail": exc.detail}, {}
        status = default_status or 200
        headers: Dict[str, Any] = {}
        body = result
        if isinstance(result, tuple):
            if len(result) == 2:
                body, status = result
            elif len(result) >= 3:
                body, status, headers = result[0], result[1], result[2] or {}
        return status, body, headers


class HTTPException(Exception):
    """Simplified HTTPException compatible with FastAPI usage."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Request:  # pragma: no cover - simple placeholder for typing
    """Placeholder request type for handlers expecting FastAPI Request."""


def Form(default: Any = ..., **_kwargs: Any) -> Any:  # pragma: no cover - placeholder dependency
    """Return default value to emulate FastAPI Form dependency."""

    return default


__all__ = ["FastAPI", "HTTPException", "Request", "Form"]
