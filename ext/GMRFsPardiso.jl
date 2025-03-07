# COV_EXCL_START
module GMRFsPardiso

using GMRFs
using Pardiso
using Random, SparseArrays, LinearAlgebra, Distributions, LinearMaps

_ensure_dense(x) = x
_ensure_dense(x::SparseVector) = Array(x)

abstract type AbstractPardisoGMRFSolver <: AbstractSolver end

_has_factorization(s::AbstractPardisoGMRFSolver) = s.ps.iparm[18] > 0
function _factorize_if_necessary(s::AbstractPardisoGMRFSolver)
    if !_has_factorization(s)
        Pardiso.set_phase!(s.ps, Pardiso.ANALYSIS_NUM_FACT)
        x = Array{Float64}(undef, size(s.Q_tril, 1))
        Pardiso.pardiso(s.ps, x, s.Q_tril, x)
    end
end
function _prepare_for_solve(s::AbstractPardisoGMRFSolver)
    if !_has_factorization(s)
        Pardiso.set_phase!(s.ps, Pardiso.ANALYSIS_NUM_FACT_SOLVE_REFINE)
    else
        Pardiso.set_phase!(s.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    end
end

compute_mean(s::AbstractPardisoGMRFSolver) = s.mean
function compute_variance(s::AbstractPardisoGMRFSolver)
    if s.computed_var !== nothing
        return s.computed_var
    end
    _factorize_if_necessary(s)

    Pardiso.set_phase!(s.ps, Pardiso.SELECTED_INVERSION)
    Pardiso.set_iparm!(s.ps, 36, 1)
    B = similar(s.Q_tril)
    x = Array{Float64}(undef, size(s.Q_tril, 1))
    Pardiso.pardiso(s.ps, x, B, x)

    Pardiso.set_phase!(s.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    s.computed_var = diag(B)
    return s.computed_var
end

function compute_rand!(
    s::AbstractPardisoGMRFSolver,
    rng::Random.AbstractRNG,
    x::AbstractVector,
)
    randn!(rng, x)
    x = _ensure_dense(x)
    x = linmap_sqrt(s.precision) * x # Centered sample with covariance Q

    _prepare_for_solve(s)
    # Centered sample with covariance Q^-1
    Pardiso.pardiso(s.ps, x, s.Q_tril, copy(x))
    x .+= _ensure_dense(compute_mean(s))
    return x
end

mutable struct PardisoGMRFSolver <: AbstractPardisoGMRFSolver
    ps::PardisoSolver
    mean::AbstractVector
    precision::LinearMap
    Q_tril::SparseMatrixCSC
    computed_var::Union{Nothing,AbstractVector}

    function PardisoGMRFSolver(gmrf::AbstractGMRF)
        ps = PardisoSolver()
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.ccall_pardisoinit(ps)
        ps.msglvl = Pardiso.MESSAGE_LEVEL_ON

        Q_tril = tril(to_matrix(precision_map(gmrf)))
        new(ps, mean(gmrf), precision_map(gmrf), Q_tril, nothing)
    end
end

mutable struct LinearConditionalPardisoGMRFSolver <: AbstractPardisoGMRFSolver
    ps::PardisoSolver
    prior_mean::AbstractVector
    precision::LinearMap
    Q_tril::SparseMatrixCSC
    A::LinearMap
    Q_ϵ::LinearMap
    y::AbstractVector
    b::AbstractVector
    computed_posterior_mean::Union{Nothing,AbstractVector}
    computed_var::Union{Nothing,AbstractVector}

    function LinearConditionalPardisoGMRFSolver(gmrf::LinearConditionalGMRF)
        ps = PardisoSolver()
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        Pardiso.ccall_pardisoinit(ps)
        ps.msglvl = Pardiso.MESSAGE_LEVEL_ON

        Q_tril = tril(to_matrix(precision_map(gmrf)))
        new(
            ps,
            mean(gmrf.prior),
            precision_map(gmrf),
            Q_tril,
            gmrf.A,
            gmrf.Q_ϵ,
            gmrf.y,
            gmrf.b,
            nothing,
            nothing,
        )
    end
end

function compute_mean(s::LinearConditionalPardisoGMRFSolver)
    if s.computed_posterior_mean !== nothing
        return s.computed_posterior_mean
    end

    _prepare_for_solve(s)
    μ = _ensure_dense(s.prior_mean)
    residual = _ensure_dense(s.y - (s.A * μ + s.b))
    rhs = _ensure_dense(s.A' * (s.Q_ϵ * residual))
    x = similar(rhs)
    Pardiso.pardiso(s.ps, x, s.Q_tril, rhs)
    s.computed_posterior_mean = μ + x
    return s.computed_posterior_mean
end

function GMRFs.construct_solver(_::PardisoGMRFSolverBlueprint, gmrf::AbstractGMRF)
    return PardisoGMRFSolver(gmrf)
end

function GMRFs.construct_solver(_::PardisoGMRFSolverBlueprint, gmrf::LinearConditionalGMRF)
    return LinearConditionalPardisoGMRFSolver(gmrf)
end

end
# COV_EXCL_STOP
