"""
visualization/__init__.py
--------------------------
Shared utilities for all visualization modules.

Every plot module calls:
    from visualization import _save
    _save(fig, out_dir, figname)

which saves as both PDF (for the paper) and PNG (for quick review).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _save(
    fig,
    out_dir: Path,
    figname: str,
    dpi: int = 300,
    formats: tuple = ("pdf", "png"),
) -> None:
    """
    Save a matplotlib figure to out_dir/{figname}.{fmt} for each format.

    Parameters
    ----------
    fig      Matplotlib Figure object.
    out_dir  Output directory (created if it doesn't exist).
    figname  Base filename without extension.
    dpi      Resolution for raster formats (PNG).
    formats  Tuple of output formats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = out_dir / f"{figname}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi if fmt != "pdf" else None,
            bbox_inches="tight",
        )

    plt.close(fig)
    print(f"  Saved → {out_dir}/{figname}.{{{'|'.join(formats)}}}")