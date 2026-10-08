"""Command-line wrapper for the native FLASH THEIA adapter."""

from __future__ import annotations

import sys

from flash_e3 import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(("--dataset", "theia"))
    main()
