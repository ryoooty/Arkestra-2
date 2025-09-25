"""Minimal YAML stub using json for tests."""

from __future__ import annotations

import json
from typing import Any

from app.util import simple_yaml


def safe_load(data: str) -> Any:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return simple_yaml.loads(data)
