# GaussianMarkovRandomFields.jl Benchmarks

This directory contains benchmark scripts for evaluating the performance of GaussianMarkovRandomFields.jl.

## Setup

The benchmarks have their own environment to avoid polluting the main package environment:

```bash
cd benchmarks
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Benchmarks

### Autodiff Comparison

Compares FiniteDiff, Zygote, and Enzyme for computing gradients through a realistic GMRF workflow with ~100 hyperparameters:

```bash
cd benchmarks
julia --project=. autodiff_comparison.jl
```

This benchmark:
- Uses a RW1 temporal model with 100 time points
- Optimizes 101 hyperparameters (100 mean parameters + 1 precision)
- Tests the full workflow: hyperparameters → GMRF → gaussian_approximation → logpdf
- Reports timing, allocations, memory usage, and gradient accuracy

### Regression Suite (CI)

`benchmarks.jl` defines a `BenchmarkTools.BenchmarkGroup` named `SUITE` that
exercises the most user-facing operations of the package on small but
representative workloads. It is intended to run in CI on every pull request
to detect performance regressions early.

Run locally:

```bash
cd benchmarks
julia --project=. run_regression.jl              # Print summary
julia --project=. run_regression.jl results.json # Also save raw JSON
```

The suite currently covers:

- **`latent_models/`** — `RW1Model` / `RW2Model` precision-matrix assembly,
  `BesagModel` construction on a 20×20 grid.
- **`gmrf/`** — `GMRF` construction (LDLt and CHOLMOD), `logpdf`, `var`
  (selected inversion), `rand` (backward solve).
- **`gaussian_approximation/`** — Fisher-scoring loop with Poisson
  observations over an RW1 prior (n=500).
- **`autodiff/`** — ForwardDiff and Zygote gradients through the full
  hyperparameters → GMRF → `gaussian_approximation` → `logpdf` pipeline.
  Tracked separately because forward- and reverse-mode exercise largely
  disjoint code paths.

Comparing two runs:

```bash
julia --project=. compare_results.jl baseline.json pr.json report.md
```

The script uses `BenchmarkTools.judge` on the median trial time. A 50%
slowdown on any benchmark is flagged as a regression and the script exits
non-zero. CI surfaces the comparison as a PR comment via the `Benchmark`
workflow.

## Adding New Benchmarks

For the regression suite, add a `@benchmarkable` entry to `benchmarks.jl`
under one of the existing groups (or create a new group). Keep individual
workloads small — the whole suite should stay under a couple of minutes on
a GitHub Actions runner.

For ad-hoc comparisons, create a new `*_comparison.jl` file in this
directory using the existing scripts as templates, add any dependencies to
`Project.toml`, and document it above.
