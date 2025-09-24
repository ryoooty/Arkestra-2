"""Minimal FastAPI stub for tests without external dependency."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

RouteKey = Tuple[str, str]


class HTTPException(Exception):
    """Simplified HTTP exception matching the FastAPI interface used in tests."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class Route:
    handler: Callable[..., Any]
    default_status: int


class Request:  # pragma: no cover - only used for typing
    def __init__(self, **kwargs: Any) -> None:
        self.scope = kwargs


def Form(default: Any = ...) -> Any:  # pragma: no cover - dependency placeholder
    return default


class FastAPI:
    """Very small subset of FastAPI used in unit tests."""

    def __init__(self, title: str | None = None, version: str | None = None) -> None:
        self.title = title or "FastAPI"
        self.version = version or "0.0"
        self._routes: Dict[RouteKey, Route] = {}
        self._startup_handlers: list[Callable[[], Any]] = []

    def _add_route(self, method: str, path: str, *, status_code: Optional[int] = None, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[(method.upper(), path)] = Route(func, status_code or 200)
            return func

        return decorator

    def on_event(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if event != "startup":
            raise ValueError("Only 'startup' event is supported in the stub")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._startup_handlers.append(func)
            return func

        return decorator

    def _run_startup(self) -> None:
        for handler in self._startup_handlers:
            handler()

    def middleware(self, middleware_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def get(self, path: str, *, status_code: Optional[int] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("GET", path, status_code=status_code, **kwargs)

    def post(self, path: str, *, status_code: Optional[int] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("POST", path, status_code=status_code, **kwargs)

    def put(self, path: str, *, status_code: Optional[int] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("PUT", path, status_code=status_code, **kwargs)

    def delete(self, path: str, *, status_code: Optional[int] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("DELETE", path, status_code=status_code, **kwargs)

    def _dispatch(self, method: str, path: str, **kwargs: Any) -> Tuple[int, Any, Dict[str, Any]]:
        key = (method.upper(), path)
        route = self._routes.get(key)
        if route is None:
            raise KeyError(f"Route not found for {method} {path}")

        call_kwargs = dict(kwargs)
        if "json" in call_kwargs:
            json_payload = call_kwargs.pop("json")
            params = list(inspect.signature(route.handler).parameters.values())
            if params:
                first = params[0]
                call_kwargs[first.name] = json_payload

        try:
            result = route.handler(**call_kwargs)
        except HTTPException as exc:  # pragma: no cover - trivial
            return exc.status_code, {"detail": exc.detail}, {}

        status = route.default_status
        headers: Dict[str, Any] = {}
        body = result
        if isinstance(result, tuple):
            if len(result) == 2:
                body, status = result
            elif len(result) >= 3:
                body, status, headers = result[0], result[1], result[2] or {}
        return status, body, headers


__all__ = ["FastAPI", "HTTPException", "Request", "Form"]
