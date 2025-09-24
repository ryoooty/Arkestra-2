"""
Generate a draft spec for a new tool based on name+purpose.
"""

import sys
import json
from app.core.llm import generate as llm_generate


def gen_spec(name: str, purpose: str):
    prompt = f"""
Ты — помощник-архитектор инструментов Arkestra.
Создай YAML-черновик для нового инструмента.

Имя: {name}
Назначение: {purpose}

Формат:
name: "{name}"
title: "..."
description: "..."
instruction: |
  (как senior должен вызывать инструмент, какие аргументы и что вернётся)
entrypoint: "app.tools.{name}:main"
enabled: true
"""
    out = llm_generate("senior", prompt, max_new_tokens=300)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/gen_tool_spec.py <name> <purpose>")
        sys.exit(1)
    gen_spec(sys.argv[1], " ".join(sys.argv[2:]))
