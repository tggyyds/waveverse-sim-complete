"""Minimal runner so the test files work without pytest (the test_* functions are also pytest-compatible)."""
import json
import sys
import time
import traceback
from pathlib import Path


def run(module_globals, record_to=None):
    fails, rows = 0, []
    for name, fn in sorted(module_globals.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name, flush=True)
                rows.append(dict(test=name, verdict="PASS"))
            except Exception as e:
                fails += 1
                print("FAIL", name, flush=True)
                traceback.print_exc()
                rows.append(dict(test=name, verdict="FAIL", error=repr(e)))
    if record_to:
        Path(record_to).write_text(json.dumps(dict(tests=rows, created=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
    print("failures", fails)
    sys.exit(1 if fails else 0)
