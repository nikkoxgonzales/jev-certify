"""Shared test setup: importable package, and a writable tmp dir.

The ``tmp_path`` fixture is redirected into the project because some environments
(locked-down Windows profiles, sandboxes) deny access to the system temp directory,
which makes every fixture-based test error out before it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest hook signature
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(ROOT / ".pytest-tmp")
