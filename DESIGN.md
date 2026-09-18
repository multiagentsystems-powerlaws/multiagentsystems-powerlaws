# Experimental Design — Power Laws of Collective Cognition in LLM MAS

Locked-in configuration for COLM 2026 submission.

## Agent scales
N ∈ {8, 16, 32, 64, 128}

5 scale points, 16× span (log₁₀ = 1.1 — one full decade).
N=128 is mandatory: it is the anchor that makes Figure 5
(x_max ~ N^β) look like a real scaling law rather than an interesting trend.

If runtime forces cuts, reduce tasks before dropping N=128.

## Seeds
3 seeds per (benchmark × topology × scale × task) combination.
Gives 210 runs per scale point (70 tasks × 3 seeds).
SE of x_max estimate ≈ 0.069 × x_max — well-constrained.

## Tasks
40 tasks total:
  MARBLE (MultiAgentBench):  10  (3 easy,  4 medium,  3 hard)
  GAIA:                      10  (3 easy,  3 medium,  4 hard)
  REALM-Bench:               10  (2 easy,  4 medium,  4 hard)
  SWE-bench Verified:        10  (1 easy,  3 medium,  4 hard) — harder bias for depth

## Topologies
All 7:
  chain, star, tree, full_mesh, sparse_mesh, hybrid_modular, dynamic_reputation

## Statistical power rationale

### H1 (power laws exist)
At N=8, 70 tasks × 3 seeds = 210 runs.
Tail samples (n_tail ≥ 100) met for all topologies except star's merge fan-in.
Star's lower statistical power is expected (structurally constrained) and
should be reported explicitly; star must not be load-bearing for H1.

### H3 (x_max ~ N^β superlinear scaling)
5 scale points over 16× span.
At α ≈ 2.3, Newman Eq. 23 predicts x_max at N=128 ≈ 10× larger than N=8.
This is Figure 5 — the paper's headline quantitative result.
4 points (N=8..64) is the minimum; 5 points (including N=128) is where
the slope becomes visually unambiguous and the error bars are tight.

## Total sweep statistics
  Total runs:    5 scales × 7 topologies × 40 tasks × 3 seeds = 4,200
  Total events:  ~603K
  Est. tokens:   ~483M
  Est. cost:     ~$386  (gpt-4o-mini, blended $0.80/1M)
  Wall time:     ~24h   (4 parallel Slurm jobs)

## Running the sweep
  cd ~/mas-powerlaws
  mkdir -p logs
  bash slurm/submit_all.sh

## After the sweep
  python scripts/run_visualizations.py \
      --data-roots /scratch/kavanav/mas-results/MARBLE \
                   /scratch/kavanav/mas-results/GAIA \
                   /scratch/kavanav/mas-results/REALM \
                   /scratch/kavanav/mas-results/SWE-bench \
      --out figures/main

  python scripts/generate_tables.py \
      --data-roots /scratch/kavanav/mas-results/{MARBLE,GAIA,REALM,SWE-bench} \
      --out tables/
