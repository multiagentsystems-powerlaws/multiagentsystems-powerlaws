#!/usr/bin/env python3
"""
scripts/diagnose_benchmarks.py
--------------------------------
Inspects the actual file/module structure of MARBLE and REALM-Bench
on this machine and prints exactly what keys/attributes are present.

Run this FIRST when benchmark loading fails.
Output tells you exactly what to fix in the wrapper.

Usage
-----
    python scripts/diagnose_benchmarks.py
    python scripts/diagnose_benchmarks.py --benchmark MARBLE
    python scripts/diagnose_benchmarks.py --benchmark REALM
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bold(s):  return f"\033[1m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"


# ─────────────────────────────────────────────────────────────────────────
# MARBLE diagnostics
# ─────────────────────────────────────────────────────────────────────────

def diagnose_marble():
    print(bold("\n═══ MARBLE diagnostics ═══"))

    # 1. Find the module
    marble_candidates = [
        Path("/home/kavanav/MARBLE"),
        Path("/home/kavanav/mas-powerlaws/MARBLE"),
        ROOT / "MARBLE",
    ]
    marble_root = None
    for c in marble_candidates:
        if c.exists():
            marble_root = c
            print(green(f"  Found MARBLE directory: {c}"))
            break
    if marble_root is None:
        print(red("  MARBLE directory not found at any of:"))
        for c in marble_candidates:
            print(f"    {c}")
        print("\n  Fix: set MARBLE_PATH in src/benchmark_wrappers/marble.py")
        return

    # 2. Directory structure
    print(f"\n  Top-level contents of {marble_root}:")
    items = sorted(marble_root.iterdir())
    for item in items[:20]:
        t = "DIR " if item.is_dir() else "FILE"
        size = f"{item.stat().st_size:>8,}b" if item.is_file() else ""
        print(f"    {t}  {item.name:<40} {size}")
    if len(items) > 20:
        print(f"    ... and {len(items)-20} more")

    # 3. Try importing as Python module
    print(f"\n  Trying: import MARBLE")
    try:
        if str(marble_root.parent) not in sys.path:
            sys.path.insert(0, str(marble_root.parent))
        import MARBLE as m
        print(green(f"  Import succeeded"))
        attrs = [a for a in dir(m) if not a.startswith('_')]
        print(f"  Top-level attributes: {attrs[:30]}")

        # Check for common task-access patterns
        for attr in ['get_tasks', 'tasks', 'TASKS', 'load', 'dataset',
                      'get_all_tasks', 'load_tasks', 'data']:
            if hasattr(m, attr):
                val = getattr(m, attr)
                print(green(f"  Found: MARBLE.{attr} = {type(val).__name__}"), end="")
                if callable(val):
                    print(" (callable)")
                    try:
                        result = val()
                        print(f"    Calling it returns: {type(result).__name__}, len={len(result) if hasattr(result,'__len__') else '?'}")
                        if result:
                            first = result[0] if hasattr(result,'__getitem__') else next(iter(result))
                            print(f"    First item type: {type(first).__name__}")
                            if isinstance(first, dict):
                                print(f"    First item keys: {list(first.keys())}")
                                _show_sample(first)
                    except Exception as e:
                        print(f"    Call failed: {e}")
                elif isinstance(val, (list, tuple)) and val:
                    print(f"  len={len(val)}")
                    first = val[0]
                    if isinstance(first, dict):
                        print(f"    First item keys: {list(first.keys())}")
                        _show_sample(first)
                else:
                    print()
    except ImportError as e:
        print(yellow(f"  Import failed: {e}"))
        print("  Will use file-based fallback")

    # 4. Search for JSON/JSONL task files
    print(f"\n  Searching for JSON/JSONL files in {marble_root}:")
    json_files = list(marble_root.rglob("*.json")) + list(marble_root.rglob("*.jsonl"))
    json_files = sorted(json_files)[:15]
    if not json_files:
        print(red("  No JSON/JSONL files found"))
    for f in json_files:
        size = f.stat().st_size
        print(f"    {f.relative_to(marble_root)}  ({size:,}b)")
        if size < 10_000_000:  # < 10MB, safe to peek
            _peek_json_file(f)

    # 5. Search for CSV files as alternative
    csv_files = list(marble_root.rglob("*.csv"))[:5]
    if csv_files:
        print(f"\n  CSV files found:")
        for f in csv_files:
            print(f"    {f.relative_to(marble_root)}  ({f.stat().st_size:,}b)")
            _peek_csv_file(f)


# ─────────────────────────────────────────────────────────────────────────
# REALM diagnostics
# ─────────────────────────────────────────────────────────────────────────

def diagnose_realm():
    print(bold("\n═══ REALM-Bench diagnostics ═══"))

    realm_candidates = [
        ROOT / "REALM-Bench" / "datasets",
        Path("/home/kavanav/mas-powerlaws/REALM-Bench/datasets"),
        Path("/home/kavanav/REALM-Bench/datasets"),
        ROOT / "REALM-Bench",
    ]

    realm_root = None
    for c in realm_candidates:
        if c.exists():
            realm_root = c
            print(green(f"  Found REALM directory: {c}"))
            break
    if realm_root is None:
        print(red("  REALM directory not found at any of:"))
        for c in realm_candidates:
            print(f"    {c}")
        return

    # Directory structure
    print(f"\n  Top-level contents:")
    for item in sorted(realm_root.iterdir())[:20]:
        t = "DIR " if item.is_dir() else "FILE"
        size = f"{item.stat().st_size:>8,}b" if item.is_file() else ""
        print(f"    {t}  {item.name:<40} {size}")

    # All JSON/JSONL files
    print(f"\n  All JSON/JSONL files under {realm_root}:")
    all_files = sorted(
        list(realm_root.rglob("*.json")) +
        list(realm_root.rglob("*.jsonl"))
    )
    if not all_files:
        print(red("  None found"))
    for f in all_files[:20]:
        size = f.stat().st_size
        print(f"\n    {f.relative_to(realm_root)}  ({size:,}b)")
        if size < 10_000_000:
            _peek_json_file(f)


# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────

def _peek_json_file(path: Path, n_items: int = 2):
    """Print key structure of a JSON/JSONL file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".jsonl":
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not lines:
                print(f"      (empty)")
                return
            print(f"      JSONL: {len(lines)} lines")
            for line in lines[:n_items]:
                try:
                    obj = json.loads(line)
                    _show_sample(obj, indent=8)
                except Exception:
                    print(f"        (unparseable line)")
        else:
            data = json.loads(text)
            if isinstance(data, list):
                print(f"      JSON list: {len(data)} items")
                for item in data[:n_items]:
                    _show_sample(item, indent=8)
            elif isinstance(data, dict):
                print(f"      JSON dict keys: {list(data.keys())[:10]}")
                # Check if it contains a tasks array
                for k, v in data.items():
                    if isinstance(v, list) and v:
                        print(f"      data['{k}'] is list, len={len(v)}")
                        _show_sample(v[0], indent=8)
                        break
    except Exception as e:
        print(f"      (could not read: {e})")


def _show_sample(obj, indent=6):
    """Show keys + sample values of a dict."""
    if not isinstance(obj, dict):
        print(f"{' '*indent}type={type(obj).__name__}  value={str(obj)[:80]}")
        return
    print(f"{' '*indent}keys: {list(obj.keys())}")
    # Show first few values
    for k, v in list(obj.items())[:4]:
        val_str = str(v)[:80].replace('\n', ' ')
        print(f"{' '*indent}  {k!r}: {val_str!r}")


def _peek_csv_file(path: Path, n_rows: int = 2):
    """Print header + first rows of a CSV."""
    try:
        import csv
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = [row for _, row in zip(range(n_rows + 1), reader)]
        if rows:
            print(f"      CSV columns: {rows[0]}")
        if len(rows) > 1:
            print(f"      Row 1: {rows[1][:5]}")
    except Exception as e:
        print(f"      (could not read CSV: {e})")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["MARBLE","REALM","both"],
                   default="both")
    return p.parse_args()


def main():
    args = parse_args()
    print(bold("╔══════════════════════════════════╗"))
    print(bold("║   Benchmark Structure Diagnostic  ║"))
    print(bold("╚══════════════════════════════════╝"))

    if args.benchmark in ("MARBLE", "both"):
        diagnose_marble()

    if args.benchmark in ("REALM", "both"):
        diagnose_realm()

    print(bold("\n═══ What to do with this output ═══"))
    print("""
  1. Paste the output back to Claude.
  2. Claude will rewrite marble.py and realm_bench.py
     to match the actual file structure found above.
  3. Re-run: python scripts/test_pipeline.py --no-llm
""")


if __name__ == "__main__":
    main()
