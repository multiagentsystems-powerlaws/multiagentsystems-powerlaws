#!/usr/bin/env python3
"""
main.py — repo-root convenience entry point.

Equivalent to `python scripts/main.py ...`. See scripts/main.py for full
pipeline documentation (stages: sweep -> extract -> analyse -> tables ->
figures) and scripts/run_sweep.py / scripts/run_analysis.py /
scripts/run_visualizations.py / scripts/generate_tables.py for the
distributed-cluster workflow (see slurm/).

Usage:
    python main.py --mode smoke --data-root data/smoke
    python main.py --mode full --benchmark GAIA --data-root data/runs/GAIA
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from main import main  # noqa: E402  (scripts/main.py)

if __name__ == "__main__":
    asyncio.run(main())
