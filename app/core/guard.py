import re, yaml
from pathlib import Path
from typing import Tuple, Dict

_cfg = yaml.safe_load(Path("config/guard.yaml").read_text(encoding="utf-8"))

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
_BAD_PATTERNS = [re.compile(p) for p in _cfg.get("profanities", [])]
_MASK = _cfg.get("replacements",{}).get("profanity_mask","***")

def _soften_word(m: re.Match) -> str:
    w = m.group(0)
    if len(w)<=2: return _MASK
    return w[0] + _MASK + w[-1]

def soft_censor(text: str) -> Tuple[str, Dict]:
    hits = {"profanity":0, "pii":0}
    out = text

    for regex in _BAD_PATTERNS:
        out, n = regex.subn(_soften_word, out)
        hits["profanity"] += n

    if _cfg.get("mask_pii",{}).get("email", True):
        out, n = _EMAIL.subn("[email скрыт]", out); hits["pii"] += n
    if _cfg.get("mask_pii",{}).get("phone", True):
        out, n = _PHONE.subn("[номер скрыт]", out); hits["pii"] += n

    if hits["profanity"]>0 and _cfg.get("style",{}).get("soften_ack", True):
        out += " (пара слов зашифрованы → ***)"
    return out, hits
