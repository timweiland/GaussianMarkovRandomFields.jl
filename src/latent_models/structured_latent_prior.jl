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
    StructuredLatentPrior{G, P, HN, C} <: NonGaussianLatentPrior

A non-Gaussian latent prior expressed as a *factor graph*: a tuple of [`LatentFactorGroup`](@ref)s
whose per-factor log-densities sum to `log p(x | θ)`. The gradient and **sparse** Hessian are
assembled by differentiating each group's small factor function and scattering the local
contributions directly into the global structures — so the heavy machinery is generic, the only
model-specific code AD specialises on is the handful of tiny factor functions, and the work per
Newton iterate is O(nnz), not O(n²).

The factor-structured counterpart of [`AutoDiffLatentPrior`](@ref) (which differentiates one
opaque whole-model closure). It carries the same `NonGaussianLatentPrior` interface
(`local_quadratic`, `prior_logdensity`, the IFT hooks), so the GA / IFT / marginal-likelihood
pipeline treats it identically.

# Fields
- `n`: number of latent variables.
- `groups`: tuple of `LatentFactorGroup`s.
- `pattern`: structural Hessian sparsity (typically the prior∪observation union), so the
  per-iterate `Q` aligns positionally with a reused `GMRFWorkspace` factorisation.
- `posmaps`: per-group, per-factor `nzval` positions of the factor's `K×K` block in `pattern`
  (precomputed once so assembly scatters into the sparse `Q` with no dense intermediate).
- `hyperparams`: tuple of hyperparameter names.
- `name`: model name.
- `constraints`: `nothing`, or a fixed `(A, e)` linear-equality constraint.
"""
struct StructuredLatentPrior{G <: Tuple, P <: Tuple, HN, C} <: NonGaussianLatentPrior
    n::Int
    groups::G
    pattern::SparseMatrixCSC{Bool, Int}
    posmaps::P
    hyperparams::HN
    name::Symbol
    constraints::C
end

# nzval index of entry (i, j) in the pattern CSC.
function _nzpos(pat::SparseMatrixCSC{Bool}, i::Int, j::Int)
    @inbounds for p in pat.colptr[j]:(pat.colptr[j + 1] - 1)
        pat.rowval[p] == i && return p
    end
    error("StructuredLatentPrior: Hessian entry ($i, $j) lies outside the declared pattern")
end

# Per-factor K×K block of nzval positions for a group (so the K² Hessian entries scatter
# straight into the sparse Q.nzval).
function _factor_positions(grp::LatentFactorGroup{K}, pat::SparseMatrixCSC{Bool}) where {K}
    return [
        ntuple(li -> ntuple(lj -> _nzpos(pat, vars[li], vars[lj]), Val(K)), Val(K))
            for vars in grp.vars
    ]
end

function StructuredLatentPrior(
        n::Int, groups::Tuple, pattern::SparseMatrixCSC{Bool};
        hyperparams = (), name::Symbol = :structured, constraints = nothing,
    )
    posmaps = map(g -> _factor_positions(g, pattern), groups)
    return StructuredLatentPrior(n, groups, pattern, posmaps, Tuple(hyperparams), name, constraints)
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

# --- Type-stable recursion over the heterogeneous tuple(s) of groups ---
# `for grp in groups` is type-unstable for a heterogeneous tuple; recursing on first/tail keeps
# each step specialised on the concrete group type.

# value only (line-search hook)
@inline _accum_value(::Tuple{}, x, θ, lp) = lp
@inline function _accum_value(groups::Tuple, x, θ, lp)
    lp = _value_one(first(groups), x, θ, lp)
    return _accum_value(Base.tail(groups), x, θ, lp)
end
function _value_one(grp::LatentFactorGroup{K}, x, θ, lp) where {K}
    f = grp.logp
    @inbounds for vars in grp.vars
        lp += f([x[vars[k]] for k in 1:K], θ)
    end
    return lp
end

# value + dense gradient + sparse Hessian scattered into `nzv` (scaled by `s`: -1 ⇒ Q=-H).
@inline _accum_full(::Tuple{}, ::Tuple{}, x, θ, g, nzv, s, lp) = lp
@inline function _accum_full(groups::Tuple, posmaps::Tuple, x, θ, g, nzv, s, lp)
    lp = _full_one(first(groups), first(posmaps), x, θ, g, nzv, s, lp)
    return _accum_full(Base.tail(groups), Base.tail(posmaps), x, θ, g, nzv, s, lp)
end
function _full_one(grp::LatentFactorGroup{K}, pos, x, θ, g, nzv, s, lp) where {K}
    f = grp.logp
    fθ = v -> f(v, θ)
    @inbounds for fi in eachindex(grp.vars)
        vars = grp.vars[fi]
        vals = [x[vars[k]] for k in 1:K]
        lp += f(vals, θ)
        gl = DI.gradient(fθ, DI.AutoForwardDiff(), vals)
        Hl = DI.hessian(fθ, DI.AutoForwardDiff(), vals)
        pf = pos[fi]
        for li in 1:K
            g[vars[li]] += gl[li]
            for lj in 1:K
                nzv[pf[li][lj]] += s * Hl[li, lj]
            end
        end
    end
    return lp
end

# gradient only (IFT score tangent)
@inline _accum_grad(::Tuple{}, x, θ, g) = g
@inline function _accum_grad(groups::Tuple, x, θ, g)
    _grad_one(first(groups), x, θ, g)
    return _accum_grad(Base.tail(groups), x, θ, g)
end
function _grad_one(grp::LatentFactorGroup{K}, x, θ, g) where {K}
    fθ = v -> grp.logp(v, θ)
    @inbounds for vars in grp.vars
        gl = DI.gradient(fθ, DI.AutoForwardDiff(), [x[vars[k]] for k in 1:K])
        for li in 1:K
            g[vars[li]] += gl[li]
        end
    end
    return g
end

# sparse Hessian only, scattered into `nzv` (scaled by `s`; IFT Dual posterior uses s=+1 for H)
@inline _accum_hess(::Tuple{}, ::Tuple{}, x, θ, nzv, s) = nzv
@inline function _accum_hess(groups::Tuple, posmaps::Tuple, x, θ, nzv, s)
    _hess_one(first(groups), first(posmaps), x, θ, nzv, s)
    return _accum_hess(Base.tail(groups), Base.tail(posmaps), x, θ, nzv, s)
end
function _hess_one(grp::LatentFactorGroup{K}, pos, x, θ, nzv, s) where {K}
    fθ = v -> grp.logp(v, θ)
    @inbounds for fi in eachindex(grp.vars)
        vars = grp.vars[fi]
        Hl = DI.hessian(fθ, DI.AutoForwardDiff(), [x[vars[k]] for k in 1:K])
        pf = pos[fi]
        for li in 1:K, lj in 1:K
            nzv[pf[li][lj]] += s * Hl[li, lj]
        end
    end
    return nzv
end

_pattern_csc(pat, nzv) =
    SparseMatrixCSC(pat.m, pat.n, copy(pat.colptr), copy(pat.rowval), nzv)

function local_quadratic(m::StructuredLatentPrior, x_ref::AbstractVector; θ...)
    θnt = NamedTuple(θ)
    T = _structured_eltype(x_ref, θnt)
    xS = _as_eltype(T, x_ref)
    g = zeros(T, m.n)
    nzv = zeros(T, nnz(m.pattern))
    lp = _accum_full(m.groups, m.posmaps, xS, θnt, g, nzv, -one(T), zero(T))  # Q = -H
    Q = _pattern_csc(m.pattern, nzv)
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
    return _accum_grad(m.groups, x_dual, θ_full, g)
end

function _dual_prior_hessian(
        m::StructuredLatentPrior, x_dual, x_primal, θ_full::NamedTuple, θ_primal::NamedTuple
    )
    T = eltype(x_dual)
    nzv = zeros(T, nnz(m.pattern))
    _accum_hess(m.groups, m.posmaps, x_dual, θ_full, nzv, one(T))  # H (the IFT negates downstream)
    return _pattern_csc(m.pattern, nzv)
end
