"""
visualization/error_correction_avalanche.py
---------------------------------------------
V18: Error-correction avalanche case study.
Finds a run where an early wrong claim was later corrected
via revision/contradiction cascade, and visualises the trajectory.

Shows: initial claim → contradiction burst → revision wave → corrected final answer.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from visualization._style import apply_style

CORRECTION_EVENTS = {"revise_claim", "contradict_claim", "merge_claims"}

def _find_error_correction_run(data_roots: List[Path]) -> Tuple[Optional[Path], dict]:
    """
    Find a run with:
      1. An early propose_claim
      2. A subsequent contradict_claim or revise_claim pointing to it
      3. A final merge_claims after the revision
    Returns (run_dir, analysis_dict).
    """
    best_run: Optional[Path] = None
    best_score = 0

    for root in data_roots:
        if not root.exists():
            continue
        for ep in root.rglob("events.jsonl"):
            events = []
            with open(ep) as f:
                for line in f:
                    try:
                        events.append(json.loads(line.strip()))
                    except Exception:
                        pass
            if not events:
                continue

            # Score = number of contradiction/revision events
            n_corrections = sum(
                1 for ev in events
                if ev.get("event_type") in CORRECTION_EVENTS
            )
            if n_corrections > best_score:
                best_score = n_corrections
                best_run   = ep.parent

    if best_run is None:
        return None, {}

    # Re-load and analyse
    events = []
    with open(best_run / "events.jsonl") as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass

    # Group events by phase: propose → contradict/revise → merge → finalise
    phases = {"propose": [], "correct": [], "merge": [], "finalise": []}
    for ev in events:
        etype = ev.get("event_type", "")
        if etype == "propose_claim":
            phases["propose"].append(ev)
        elif etype in ("contradict_claim", "revise_claim"):
            phases["correct"].append(ev)
        elif etype == "merge_claims":
            phases["merge"].append(ev)
        elif etype == "finalize_answer":
            phases["finalise"].append(ev)

    return best_run, {
        "events":   events,
        "phases":   phases,
        "n_agents": len(set(ev.get("agent_id","") for ev in events)),
        "n_steps":  max((ev.get("step_id") or 0 for ev in events), default=0),
        "topology": events[0].get("topology","?") if events else "?",
    }


def plot_error_correction_avalanche(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_error_correction_avalanche",
):
    apply_style()

    run_dir, analysis = _find_error_correction_run(data_roots)
    if run_dir is None:
        print("  error_correction: no suitable run found")
        return None

    events   = analysis["events"]
    phases   = analysis["phases"]
    n_steps  = analysis["n_steps"]
    topology = analysis["topology"]

    # Build timeline: step → event counts by phase
    PHASE_COLORS = {
        "propose":  "#4C72B0",
        "correct":  "#C44E52",
        "merge":    "#8172B2",
        "finalise": "#55A868",
    }
    PHASE_LABELS = {
        "propose":  "Initial proposals",
        "correct":  "Corrections (revise/contradict)",
        "merge":    "Merges/synthesis",
        "finalise": "Final answer",
    }

    step_counts: dict = {ph: defaultdict(int) for ph in PHASE_COLORS}
    for ph, evs in phases.items():
        for ev in evs:
            s = ev.get("step_id") or 0
            step_counts[ph][s] += 1

    all_steps = sorted(set(
        s for ph in step_counts for s in step_counts[ph]
    ))
    if not all_steps:
        print("  error_correction: no step data")
        return None

    fig, (ax_main, ax_timeline) = plt.subplots(
        2, 1, figsize=(7.0, 5.0),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # ── Top panel: stacked area chart of events by phase over time ──
    step_arr = np.array(all_steps)
    for ph, col in PHASE_COLORS.items():
        counts = np.array([step_counts[ph].get(s, 0) for s in all_steps])
        ax_main.fill_between(step_arr, counts, alpha=0.4, color=col,
                             label=PHASE_LABELS[ph])
        ax_main.plot(step_arr, counts, "-", color=col, lw=1.2)

    ax_main.set_xlabel("Step", fontsize=9)
    ax_main.set_ylabel("Events per step", fontsize=9)
    ax_main.set_title(
        f"Error-correction avalanche — {topology}\n"
        f"({len(events)} total events, {analysis['n_agents']} agents, "
        f"{n_steps} steps)",
        fontsize=8
    )
    ax_main.legend(fontsize=7, frameon=False, ncol=2)

    # ── Bottom panel: phase timeline bar ──
    phase_order  = ["propose", "correct", "merge", "finalise"]
    phase_sizes  = [len(phases[p]) for p in phase_order]
    cumulative   = 0
    for ph, sz in zip(phase_order, phase_sizes):
        if sz == 0:
            continue
        ax_timeline.barh(0, sz, left=cumulative, height=0.5,
                          color=PHASE_COLORS[ph], alpha=0.85,
                          edgecolor="white", lw=0.5)
        if sz > 2:
            ax_timeline.text(cumulative + sz / 2, 0,
                              ph, ha="center", va="center",
                              fontsize=6, color="white")
        cumulative += sz

    ax_timeline.set_xlim(0, cumulative)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Cumulative events", fontsize=8)
    ax_timeline.set_title("Phase composition", fontsize=8)

    fig.tight_layout()
    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
