#!/usr/bin/env python3
"""
scripts/generate_tables.py
----------------------------
Generate all 4 result tables as LaTeX + CSV.

Usage
-----
  # Full pipeline output:
  python scripts/generate_tables.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --analysis   data/analysis \\
      --extracted  data/extracted \\
      --out        tables/

  # With single-agent baseline for Table 2:
  python scripts/generate_tables.py \\
      --data-roots /scratch/.../MARBLE ... \\
      --single-agent-root /scratch/.../single_agent \\
      --analysis data/analysis --extracted data/extracted --out tables/

  # Minimal (inline extraction, no pre-computed):
  python scripts/generate_tables.py \\
      --data-roots data/runs --out tables/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Generate result tables")
    p.add_argument("--data-roots",       nargs="+", required=True)
    p.add_argument("--single-agent-root", default=None)
    p.add_argument("--analysis",          default=None,
                   help="Output dir of run_analysis.py (for pre-computed fits)")
    p.add_argument("--extracted",         default=None,
                   help="Output dir of extract_events.py (for run_metrics.csv)")
    p.add_argument("--out",               default="tables")
    return p.parse_args()


def main():
    args      = parse_args()
    roots     = [Path(r) for r in args.data_roots]
    sa_root   = Path(args.single_agent_root) if args.single_agent_root else None
    analysis  = Path(args.analysis)  if args.analysis  else None
    extracted = Path(args.extracted) if args.extracted else None
    out       = Path(args.out)

    from tables.generate_tables import generate_all_tables
    generate_all_tables(
        data_roots=roots,
        out_dir=out,
        single_agent_root=sa_root,
        analysis_dir=analysis,
        extracted_dir=extracted,
    )


if __name__ == "__main__":
    main()