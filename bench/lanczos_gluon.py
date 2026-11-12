"""Bench driver. The kernel is python/blqh/lanczos_gluon.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blqh.lanczos_gluon import main

if __name__ == "__main__":
    sys.exit(main())
