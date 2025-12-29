#!/usr/bin/env python3
"""Entry point that forwards to train.py for convenience."""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("train", run_name="__main__")


if __name__ == "__main__":
    main()
