"""Lightweight test client compatible with the FastAPI stub."""

from dataclasses import dataclass
from typing import Any


@dataclass
class _Response:
    status_code: int
    _body: Any

    def json(self) -> Any:
        return self._body


class TestClient:
    def __init__(self, app: Any) -> None:
        self.app = app
        if hasattr(app, "_run_startup"):
            app._run_startup()

    def get(self, path: str, **kwargs: Any) -> _Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> _Response:
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> _Response:
        return self._request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> _Response:
        return self._request("DELETE", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> _Response:
        try:
            status, body, _headers = self.app._dispatch(method, path, **kwargs)
        except KeyError:
            status, body = 404, {"detail": "Not Found"}
        return _Response(status_code=status, _body=body)


__all__ = ['TestClient']
TestClient.__test__ = False
