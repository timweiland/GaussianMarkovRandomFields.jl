import DifferentiationInterface as DI

export StructuredLatentPrior, LatentFactorGroup

"""
    LatentFactorGroup{K, F}

A group of conditional-prior factors that share one log-density *form*. Each factor touches `K`
latent variables; `vars[i]` lists the global latent indices of the `i`-th factor (by convention
the factor's own variable first, then its parents). `logp(vals, θ::NamedTuple) -> Real` is the
factor log-density evaluated at the local values `vals = x[vars[i]]` and hyperparameters `θ`.

Because every factor in a group calls the *same* `logp`, automatic differentiation specialises
on one small (`K`-input) function per group — not on the whole-model log-density. This is what
keeps the per-model compile cost small.
"""
struct LatentFactorGroup{K, F}
    vars::Vector{NTuple{K, Int}}
    logp::F
end

"""
    StructuredLatentPrior{G, HN, C} <: NonGaussianLatentPrior

A non-Gaussian latent prior expressed as a *factor graph*: a tuple of [`LatentFactorGroup`](@ref)s
whose per-factor log-densities sum to `log p(x | θ)`. The gradient and (sparse) Hessian are
assembled by differentiating each group's small factor function and scattering the local
contributions into the global arrays — so the heavy machinery is generic and the only
model-specific code that AD specialises on is the handful of tiny factor functions.

This is the factor-structured counterpart of [`AutoDiffLatentPrior`](@ref) (which differentiates
one opaque whole-model closure). It carries the same `NonGaussianLatentPrior` interface
(`local_quadratic`, `prior_logdensity`, the IFT hooks), so the GA / IFT / marginal-likelihood
pipeline treats it identically.

# Fields
- `n`: number of latent variables.
- `groups`: tuple of `LatentFactorGroup`s.
- `pattern`: structural Hessian sparsity (typically the prior∪observation union), so the
  per-iterate `Q` aligns positionally with a reused `GMRFWorkspace` factorisation.
- `hyperparams`: tuple of hyperparameter names.
- `name`: model name.
- `constraints`: `nothing`, or a fixed `(A, e)` linear-equality constraint.
"""
struct StructuredLatentPrior{G <: Tuple, HN, C} <: NonGaussianLatentPrior
    n::Int
    groups::G
    pattern::SparseMatrixCSC{Bool, Int}
    hyperparams::HN
    name::Symbol
    constraints::C
end

function StructuredLatentPrior(
        n::Int, groups::Tuple, pattern::SparseMatrixCSC{Bool};
        hyperparams = (), name::Symbol = :structured, constraints = nothing,
    )
    return StructuredLatentPrior(n, groups, pattern, Tuple(hyperparams), name, constraints)
end

Base.length(m::StructuredLatentPrior) = m.n
model_name(m::StructuredLatentPrior) = m.name
constraints(m::StructuredLatentPrior; kwargs...) = m.constraints
hyperparameters(m::StructuredLatentPrior) =
    NamedTuple{m.hyperparams}(ntuple(_ -> Real, length(m.hyperparams)))

# Element type for the accumulators: widen `eltype(x)` by any AD-tagged hyperparameter value
# eltypes (so an outer Dual-θ pass accumulates at the right type). Mirrors AutoDiffLatentPrior.
function _structured_eltype(x, θ::NamedTuple)
    S = promote_type(eltype(x), map(_value_eltype, values(θ))...)
    return isconcretetype(S) ? S : eltype(x)
end

# --- Type-stable recursion over the heterogeneous tuple of groups ---
# `for grp in groups` is type-unstable for a heterogeneous tuple; recursing on first/tail keeps
# each step specialised on the concrete group type.

@inline _accum_value(::Tuple{}, x, θ, lp) = lp
@inline function _accum_value(groups::Tuple, x, θ, lp)
    lp = _value_one(first(groups), x, θ, lp)
    return _accum_value(Base.tail(groups), x, θ, lp)
end
function _value_one(grp::LatentFactorGroup{K}, x, θ, lp) where {K}
    f = grp.logp
    @inbounds for vars in grp.vars
        vals = [x[vars[k]] for k in 1:K]
        lp += f(vals, θ)
    end
    return lp
end

@inline _accum_gh(::Tuple{}, x, θ, g, H, lp) = lp
@inline function _accum_gh(groups::Tuple, x, θ, g, H, lp)
    lp = _gh_one(first(groups), x, θ, g, H, lp)
    return _accum_gh(Base.tail(groups), x, θ, g, H, lp)
end
function _gh_one(grp::LatentFactorGroup{K}, x, θ, g, H, lp) where {K}
    f = grp.logp
    fθ = v -> f(v, θ)
    @inbounds for vars in grp.vars
        vals = [x[vars[k]] for k in 1:K]
        lp += f(vals, θ)
        gl = DI.gradient(fθ, DI.AutoForwardDiff(), vals)
        Hl = DI.hessian(fθ, DI.AutoForwardDiff(), vals)
        for li in 1:K
            g[vars[li]] += gl[li]
            for lj in 1:K
                H[vars[li], vars[lj]] += Hl[li, lj]
            end
        end
    end
    return lp
end

# Same scatter, gradient only (the IFT score tangent).
@inline _accum_g(::Tuple{}, x, θ, g) = g
@inline function _accum_g(groups::Tuple, x, θ, g)
    _g_one(first(groups), x, θ, g)
    return _accum_g(Base.tail(groups), x, θ, g)
end
function _g_one(grp::LatentFactorGroup{K}, x, θ, g) where {K}
    fθ = v -> grp.logp(v, θ)
    @inbounds for vars in grp.vars
        vals = [x[vars[k]] for k in 1:K]
        gl = DI.gradient(fθ, DI.AutoForwardDiff(), vals)
        for li in 1:K
            g[vars[li]] += gl[li]
        end
    end
    return g
end

# Same scatter, Hessian only (the IFT Dual posterior precision).
@inline _accum_h(::Tuple{}, x, θ, H) = H
@inline function _accum_h(groups::Tuple, x, θ, H)
    _h_one(first(groups), x, θ, H)
    return _accum_h(Base.tail(groups), x, θ, H)
end
function _h_one(grp::LatentFactorGroup{K}, x, θ, H) where {K}
    fθ = v -> grp.logp(v, θ)
    @inbounds for vars in grp.vars
        vals = [x[vars[k]] for k in 1:K]
        Hl = DI.hessian(fθ, DI.AutoForwardDiff(), vals)
        for li in 1:K, lj in 1:K
            H[vars[li], vars[lj]] += Hl[li, lj]
        end
    end
    return H
end

# Restrict a dense matrix onto the structural pattern, returning a CSC whose colptr/rowval are
# the pattern's (so it lines up positionally with the workspace factor across iterates).
function _onto_pattern(M, pat::SparseMatrixCSC{Bool})
    V = Vector{eltype(M)}(undef, nnz(pat))
    k = 0
    @inbounds for j in 1:pat.n, p in pat.colptr[j]:(pat.colptr[j + 1] - 1)
        k += 1
        V[k] = M[pat.rowval[p], j]
    end
    return SparseMatrixCSC(pat.m, pat.n, copy(pat.colptr), copy(pat.rowval), V)
end

function local_quadratic(m::StructuredLatentPrior, x_ref::AbstractVector; θ...)
    θnt = NamedTuple(θ)
    T = _structured_eltype(x_ref, θnt)
    xS = _as_eltype(T, x_ref)
    g = zeros(T, m.n)
    H = zeros(T, m.n, m.n)
    lp = _accum_gh(m.groups, xS, θnt, g, H, zero(T))
    Q = _onto_pattern(-H, m.pattern)
    return LocalLatentQuadratic(Q, g + Q * xS, lp, xS)
end

function prior_logdensity(m::StructuredLatentPrior, x::AbstractVector; θ...)
    θnt = NamedTuple(θ)
    T = _structured_eltype(x, θnt)
    return _accum_value(m.groups, _as_eltype(T, x), θnt, zero(T))
end

# IFT hooks: assemble the latent gradient / Hessian at a Dual-valued x from the small per-factor
# derivatives (each nests cleanly under the outer Dual).
function _dual_prior_gradient(m::StructuredLatentPrior, x_dual, θ_full::NamedTuple)
    g = zeros(eltype(x_dual), m.n)
    return _accum_g(m.groups, x_dual, θ_full, g)
end

function _dual_prior_hessian(
        m::StructuredLatentPrior, x_dual, x_primal, θ_full::NamedTuple, θ_primal::NamedTuple
    )
    H = zeros(eltype(x_dual), m.n, m.n)
    _accum_h(m.groups, x_dual, θ_full, H)
    return _onto_pattern(H, m.pattern)
end
