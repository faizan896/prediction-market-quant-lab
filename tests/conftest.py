"""Make `pytest` work in a fresh clone without installing the package first.

`pip install -e ".[dev]"` is still the documented setup, but a plain `pytest`
in a checkout should not fail with `ModuleNotFoundError: quantlab.data` — and
worse, it can silently pick up a *different* `quantlab` from site-packages and
test the wrong code. Prepending the repo root guarantees the tests exercise the
working tree they live in.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
