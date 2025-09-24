"""Very small YAML-like parser for test fixtures."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, List


def load(path: str | Path) -> Any:
    return loads(Path(path).read_text(encoding='utf-8'))


def loads(text: str) -> Any:
    lines: List[str] = []
    for raw in text.splitlines():
        stripped = raw.split('#', 1)[0].rstrip()
        if not stripped:
            continue
        for part in _expand_semicolons(stripped):
            lines.append(part)

    parser = _Parser(lines)
    return parser.parse_block(0)


def _expand_semicolons(line: str) -> Iterable[str]:
    if ';' not in line:
        yield line
        return
    indent = len(line) - len(line.lstrip(' '))
    prefix = ' ' * indent
    for chunk in line.strip().split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        yield prefix + chunk


class _Parser:
    def __init__(self, lines: List[str]) -> None:
        self.lines = lines
        self.index = 0

    def parse_block(self, indent: int) -> Any:
        items: dict[str, Any] = {}
        array: list[Any] | None = None
        while self.index < len(self.lines):
            line = self.lines[self.index]
            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indent at line: {line!r}")
            content = line.strip()
            if content.startswith('- '):
                if items:
                    raise ValueError('Cannot mix list and dict at same level')
                if array is None:
                    array = []
                value_text = content[2:].strip()
                self.index += 1
                if value_text:
                    array.append(_parse_value(value_text))
                else:
                    array.append(self.parse_block(indent + 2))
            else:
                if array is not None:
                    raise ValueError('Cannot mix list and dict at same level')
                key, _, value_text = content.partition(':')
                key = key.strip()
                value_text = value_text.strip()
                self.index += 1
                if not value_text:
                    value = self.parse_block(indent + 2)
                else:
                    value = _parse_value(value_text)
                items[key] = value
        return array if array is not None else items

def _parse_value(text: str) -> Any:
    if text.startswith('{') and text.endswith('}'):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for part in inner.split(','):
            if not part.strip():
                continue
            key, _, value_text = part.partition(':')
            if not _:
                raise ValueError(f"Invalid inline mapping: {part!r}")
            result[key.strip()] = _parse_value(value_text.strip())
        return result
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part.strip()) for part in inner.split(',') if part.strip()]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return ast.literal_eval(text)
    lowered = text.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    try:
        if '.' in text:
            return float(text)
        return int(text)
    except ValueError:
        return text
