import logging
import sys
import time
from contextlib import contextmanager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("arkestra")


@contextmanager
def span(name: str):
    t0 = time.time()
    log.info(f"[start] {name}")
    try:
        yield
        log.info(f"[done] {name} dt={time.time() - t0:.3f}s")
    except Exception as e:  # pragma: no cover - logging side-effect
        log.exception(f"[fail] {name}: {e}")
        raise
