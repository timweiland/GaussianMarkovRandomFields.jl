#!/usr/bin/env julia
"""
Performance regression benchmark suite for GaussianMarkovRandomFields.jl.

Defines a `BenchmarkTools.BenchmarkGroup` named `SUITE` that exercises
representative-but-small workloads across the package's most user-facing
operations. The aim is to catch performance regressions in CI without
running large jobs.

Conventions:
- File path / variable name follow the `PkgBenchmark.jl` convention so the
  suite can be picked up by `PkgBenchmark.benchmarkpkg` and similar tools.
- Each top-level subgroup corresponds to one functional area of the package.
- Problem sizes are tuned so the full suite runs in roughly a minute on a
  laptop. Larger workloads belong in `*_comparison.jl` scripts.

Usage (standalone):
    cd benchmarks
    julia --project=. -e 'include("benchmarks.jl"); using BenchmarkTools; results = run(SUITE; verbose=true); display(results)'

Usage (via PkgBenchmark):
    julia --project=benchmarks -e 'using PkgBenchmark; result = benchmarkpkg("GaussianMarkovRandomFields"); export_markdown(stdout, result)'
"""

using GaussianMarkovRandomFields
using BenchmarkTools
using Distributions: Poisson, logpdf
using SparseArrays
using LinearAlgebra
using LinearSolve
using Random
using ForwardDiff
using Zygote
using DifferentiationInterface: DifferentiationInterface, AutoForwardDiff, AutoZygote

const SUITE = BenchmarkGroup()

# ---------------------------------------------------------------------------
# Deterministic data shared across groups
# ---------------------------------------------------------------------------

Random.seed!(20260525)

# Sizes chosen to be representative without being heavy.
const N_SMALL = 500   # RW1/RW2/Besag baseline
const N_MED = 1000    # GMRF core operations
const N_AD = 100      # Autodiff pipeline (matches existing autodiff bench)

# Pre-built precision matrices used by several groups.
# RW1Model returns SymTridiagonal, which is what LDLtFactorization expects.
# Convert to sparse only when feeding CHOLMOD.
const Q_RW1_MED = precision_matrix(RW1Model(N_MED); τ = 1.0)
const Q_RW1_SMALL = precision_matrix(RW1Model(N_SMALL); τ = 1.0)

# Mean / evaluation vectors.
const MU_MED = zeros(N_MED)
const X_MED = randn(N_MED)
const MU_SMALL = zeros(N_SMALL)

# Pre-built GMRFs reused across groups so we benchmark the operation, not the
# construction cost (construction itself is benchmarked separately).
const GMRF_MED_LDLT = GMRF(MU_MED, Q_RW1_MED, LinearSolve.LDLtFactorization())
const GMRF_MED_CHOLMOD = GMRF(MU_MED, sparse(Q_RW1_MED), LinearSolve.CHOLMODFactorization())

# Poisson observations for gaussian_approximation benchmarks.
const POISSON_LATENT_SMALL = 0.5 .* sin.(range(0, 2π; length = N_SMALL))
const Y_POISSON_SMALL = PoissonObservations(
    rand.(MersenneTwister(1), Poisson.(exp.(POISSON_LATENT_SMALL .+ 0.5)))
)
const OBS_LIK_POISSON_SMALL = ExponentialFamily(Poisson)(Y_POISSON_SMALL)
const GMRF_PRIOR_SMALL = GMRF(MU_SMALL, Q_RW1_SMALL, LinearSolve.LDLtFactorization())

# Adjacency matrix for a small 2D grid (Besag spatial model).
function _grid_adjacency(nx::Int, ny::Int)
    n = nx * ny
    I = Int[]
    J = Int[]
    idx(i, j) = (j - 1) * nx + i
    for j in 1:ny, i in 1:nx
        k = idx(i, j)
        if i < nx
            push!(I, k); push!(J, idx(i + 1, j))
            push!(I, idx(i + 1, j)); push!(J, k)
        end
        if j < ny
            push!(I, k); push!(J, idx(i, j + 1))
            push!(I, idx(i, j + 1)); push!(J, k)
        end
    end
    return sparse(I, J, ones(Bool, length(I)), n, n)
end

const BESAG_ADJ = _grid_adjacency(20, 20)  # 400 nodes, mirrors typical spatial use cases

# Autodiff pipeline data (matches the existing autodiff_comparison.jl shape).
let
    Random.seed!(123)
    μ_true = 0.5 .+ 0.3 .* sin.(range(0, 2π; length = N_AD))
    τ_true = 5.0
    θ_init = vcat(μ_true, log(τ_true)) .+ randn(N_AD + 1) .* 0.1
    x_true = μ_true .+ cumsum(randn(N_AD)) .* sqrt(1 / τ_true) .* 0.5
    x_true .-= sum(x_true) / N_AD
    y_counts = rand.(Poisson.(exp.(x_true .+ 0.5)))
    y_obs = PoissonObservations(y_counts)
    x_eval = randn(N_AD) .+ 0.3
    global const AD_THETA = θ_init
    global const AD_Y = y_obs
    global const AD_X_EVAL = x_eval
end

# Workflow used inside the autodiff benchmark.
function _ad_workflow(θ::AbstractVector, y::PoissonObservations, x_eval::AbstractVector)
    n = N_AD
    μ = θ[1:n]
    τ = exp(θ[n + 1])
    Q = precision_matrix(RW1Model(n); τ = τ)
    prior = GMRF(μ, Q, LinearSolve.LDLtFactorization())
    obs_lik = ExponentialFamily(Poisson)(y)
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, x_eval)
end

const AD_LOSS = θ -> _ad_workflow(θ, AD_Y, AD_X_EVAL)
const AD_PREP_ZYGOTE = DifferentiationInterface.prepare_gradient(AD_LOSS, AutoZygote(), AD_THETA)
const AD_PREP_FORWARDDIFF = DifferentiationInterface.prepare_gradient(AD_LOSS, AutoForwardDiff(), AD_THETA)

# ---------------------------------------------------------------------------
# 1. Latent model construction (precision matrix assembly)
# ---------------------------------------------------------------------------
SUITE["latent_models"] = BenchmarkGroup()
SUITE["latent_models"]["rw1_precision_matrix"] =
    @benchmarkable precision_matrix(model; τ = 1.0) setup = (model = RW1Model($N_MED))
SUITE["latent_models"]["rw2_precision_matrix"] =
    @benchmarkable precision_matrix(model; τ = 1.0) setup = (model = RW2Model($N_SMALL))
SUITE["latent_models"]["besag_construction"] =
    @benchmarkable BesagModel($BESAG_ADJ)
SUITE["latent_models"]["besag_instantiate"] =
    @benchmarkable model(; τ = 1.0) setup = (model = BesagModel($BESAG_ADJ))

# ---------------------------------------------------------------------------
# 2. GMRF core operations
# ---------------------------------------------------------------------------
SUITE["gmrf"] = BenchmarkGroup()

# Construction = factorization cost. Use LDLt (default for RW1) and CHOLMOD
# (default for sparse SPD precision); both are common.
SUITE["gmrf"]["construct_ldlt"] = @benchmarkable GMRF($MU_MED, $Q_RW1_MED, LinearSolve.LDLtFactorization())
SUITE["gmrf"]["construct_cholmod"] = @benchmarkable GMRF($MU_MED, $(sparse(Q_RW1_MED)), LinearSolve.CHOLMODFactorization())

# Read-only operations on a pre-built GMRF.
SUITE["gmrf"]["logpdf_ldlt"] = @benchmarkable logpdf($GMRF_MED_LDLT, $X_MED)
SUITE["gmrf"]["logpdf_cholmod"] = @benchmarkable logpdf($GMRF_MED_CHOLMOD, $X_MED)
SUITE["gmrf"]["var_selinv"] = @benchmarkable var($GMRF_MED_CHOLMOD)
SUITE["gmrf"]["rand_backward_solve"] =
    @benchmarkable rand(rng, $GMRF_MED_LDLT) setup = (rng = MersenneTwister(0))

# ---------------------------------------------------------------------------
# 3. Gaussian approximation (Fisher scoring hot path)
# ---------------------------------------------------------------------------
SUITE["gaussian_approximation"] = BenchmarkGroup()
SUITE["gaussian_approximation"]["poisson_rw1_small"] =
    @benchmarkable gaussian_approximation($GMRF_PRIOR_SMALL, $OBS_LIK_POISSON_SMALL)

# ---------------------------------------------------------------------------
# 4. Autodiff pipeline — gradient through full workflow.
# Forward- and reverse-mode are tracked separately because they exercise
# largely disjoint code paths (custom ChainRules vs. ForwardDiff dual numbers).
# ---------------------------------------------------------------------------
SUITE["autodiff"] = BenchmarkGroup()
SUITE["autodiff"]["zygote_grad_full_pipeline"] =
    @benchmarkable DifferentiationInterface.gradient($AD_LOSS, $AD_PREP_ZYGOTE, AutoZygote(), $AD_THETA)
SUITE["autodiff"]["forwarddiff_grad_full_pipeline"] =
    @benchmarkable DifferentiationInterface.gradient($AD_LOSS, $AD_PREP_FORWARDDIFF, AutoForwardDiff(), $AD_THETA)

# Default tuning: shorter than BenchmarkTools defaults so the suite finishes
# quickly in CI. Override per-benchmark above when needed.
for group in values(SUITE)
    for bench in values(group)
        bench.params.seconds = 5
        bench.params.samples = 50
    end
end

SUITE
