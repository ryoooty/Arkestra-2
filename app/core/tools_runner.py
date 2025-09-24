import importlib
from typing import List, Dict


def run_all(tool_calls: List[Dict]) -> List[Dict]:
    results = []
    for call in tool_calls or []:
        name = call.get("name")
        args = call.get("args", {})
        mod_name, func_name = _resolve_entrypoint(name)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        res = fn(args)
        results.append({"name": name, "result": res})
    return results


def _resolve_entrypoint(name: str) -> tuple[str, str]:
    mapping = {
        "note.create": "app.tools.note:main",
        "reminder.create": "app.tools.reminder:main",
        "tg.message.send": "app.tools.tg_message:main",
        "messages.search_by_date": "app.tools.search_by_date:main",
        "alias.add": "app.tools.alias:add",
        "alias.set_primary": "app.tools.alias:set_primary",
    }
    ep = mapping.get(name)
    if not ep:
        raise RuntimeError(f"Unknown tool: {name}")
    mod, func = ep.split(":")
    return mod, func
