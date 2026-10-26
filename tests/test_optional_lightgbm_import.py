"""A broken optional LightGBM/Dask import must not block OnTime serving."""
import os
import subprocess
import sys
from pathlib import Path


def test_app_engine_starts_when_lightgbm_import_raises_typeerror():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import builtins
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
real_import = builtins.__import__
def broken_lightgbm_import(name, *args, **kwargs):
    if name == "lightgbm":
        raise TypeError("simulated Dask/pandas descriptor incompatibility")
    return real_import(name, *args, **kwargs)
builtins.__import__ = broken_lightgbm_import
from backend.app.engine import get_engine
engine = get_engine()
assert len(engine.queue) > 0
assert "synthetic_operational" in engine.models
from enriched_model import LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR
assert LIGHTGBM_AVAILABLE is False
assert "TypeError" in LIGHTGBM_IMPORT_ERROR
print("OK: OnTime engine starts with incompatible optional LightGBM")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "OK: OnTime engine starts" in completed.stdout
