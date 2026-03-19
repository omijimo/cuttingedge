#!/usr/bin/env python3
"""Convenience wrapper for dataset preprocessing."""
import subprocess
import sys

sys.exit(
    subprocess.call(
        [sys.executable, "-m", "accompaniment.cli", "preprocess", "--config", "configs/base.yaml"]
    )
)
