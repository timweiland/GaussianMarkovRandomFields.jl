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

## Adding New Benchmarks

1. Create a new `.jl` file in this directory
2. Use the existing `autodiff_comparison.jl` as a template
3. Add any required dependencies to `Project.toml`
4. Document the benchmark here in this README
