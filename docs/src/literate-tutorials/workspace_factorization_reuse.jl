# # Reusing factorizations across hyperparameters
#
# ## Motivation
#
# Most interesting GMRF workflows don't stop at a single evaluation. Bayesian
# hyperparameter inference (INLA, TMB-style marginal-likelihood optimization,
# HMC over hyperparameters) and MAP fitting all involve *inner loops* that
# repeatedly evaluate the same GMRF at many hyperparameter settings — each
# time computing quantities like `logpdf`, `var`, or a Gaussian approximation
# of a non-Gaussian posterior.
#
# A fresh GMRF pays the full symbolic Cholesky cost on every construction —
# fill-reducing reordering, elimination tree, supernode layout — even when
# the sparsity pattern is identical across iterations and only the numeric
# values change. For a typical inner loop that touches the prior hundreds or
# thousands of times, that overhead adds up.
#
# The `GMRFWorkspace` / `WorkspaceGMRF` machinery lets you pay the symbolic
# cost once and reuse it. This tutorial walks through a minimal example.
#
# ## Setup
#
# We use a Besag spatial model on a 100 × 100 grid (10 000 nodes) — a scale
# where symbolic factorization is meaningfully expensive but the example
# still runs quickly. A 5-point stencil graph gives realistic fill-in during
# Cholesky factorization.

using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using Random
using Printf

Random.seed!(42)

# Build a 4-neighbor adjacency matrix for an `m × n` grid.
function grid_adjacency(m::Int, n::Int)
    N = m * n
    I = Int[]
    J = Int[]
    for j in 1:n, i in 1:m
        idx = (j - 1) * m + i
        if i < m
            push!(I, idx); push!(J, idx + 1)   # vertical neighbor
        end
        if j < n
            push!(I, idx); push!(J, idx + m)   # horizontal neighbor
        end
    end
    return sparse([I; J], [J; I], 1.0, N, N)
end

m_grid = 100
W = grid_adjacency(m_grid, m_grid)
N = size(W, 1)
model = BesagModel(W)

# We'll evaluate `logpdf` at a grid of hyperparameter values, simulating the
# inner loop of hyperparameter inference.

θ_grid = [(; τ = τ) for τ in range(0.5, 2.0; length = 50)]

# Use a point satisfying the Besag sum-to-zero constraint.
z = randn(N)
z .-= sum(z) / N
nothing #hide

# ## Baseline: fresh GMRF per iteration
#
# The baseline constructs a fresh `GMRF` for each hyperparameter setting.
# Each iteration runs the full symbolic + numeric factorization pipeline.

function loop_cold(model, θ_grid, z)
    s = 0.0
    for θ in θ_grid
        gmrf = model(; θ...)
        s += logpdf(gmrf, z)
    end
    return s
end

loop_cold(model, θ_grid[1:2], z)    # warmup for compilation
s_cold = loop_cold(model, θ_grid, z)
t_cold = minimum(@elapsed(loop_cold(model, θ_grid, z)) for _ in 1:3)
@printf("Cold path: %.3f s (%d evaluations, best of 3)\n", t_cold, length(θ_grid))

# ## Warm path: one workspace, reused
#
# With `make_workspace`, the symbolic analysis is done once. Each subsequent
# call to `model(ws; θ...)` just writes new numeric values into the
# workspace's precision buffer and re-runs the numeric factorization —
# reusing the same fill-reducing reordering and supernode structure.

function loop_warm(model, θ_grid, z)
    ws = make_workspace(model; θ_grid[1]...)
    s = 0.0
    for θ in θ_grid
        gmrf = model(ws; θ...)
        s += logpdf(gmrf, z)
    end
    return s
end

loop_warm(model, θ_grid[1:2], z)    # warmup for compilation
s_warm = loop_warm(model, θ_grid, z)
t_warm = minimum(@elapsed(loop_warm(model, θ_grid, z)) for _ in 1:3)
@printf("Warm path: %.3f s (%d evaluations, best of 3)\n", t_warm, length(θ_grid))

# Both loops compute the same quantity:

@assert s_cold ≈ s_warm

# And the speedup:

@printf("Speedup: %.2fx\n", t_cold / t_warm)

# ## When this matters
#
# The workspace pattern wins whenever:
#
# - You touch a GMRF many times with a **fixed sparsity pattern** — INLA θ
#   optimization, MAP descent, HMC/NUTS over hyperparameters, cross-validation
#   grids, posterior visualization at many θ values.
# - The **symbolic factorization cost is non-negligible** relative to numeric
#   — typically true once `n` gets into the thousands, especially for models
#   with genuine 2D/3D fill-in rather than trivial tridiagonal structure.
#
# It does not help for one-off GMRF construction or workflows where the
# sparsity pattern actually changes between iterations (e.g. adaptive
# mesh refinement).
#
# ## Parallel inner loops
#
# For multi-threaded inner loops, use [`make_workspace_pool`](@ref) and the
# [`with_workspace`](@ref) RAII helper for a task-safe version of the same
# pattern. See the [Workspaces](@ref) reference for details — including the
# CHOLMOD-serializes caveat and the `CliqueTreesBackend` alternative for
# actual parallel factorization.
