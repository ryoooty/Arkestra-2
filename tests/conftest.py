import sys
from pathlib import Path

from . import yaml_stub

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure modules under test that import ``yaml`` receive the lightweight stub.
sys.modules.setdefault("yaml", yaml_stub)
