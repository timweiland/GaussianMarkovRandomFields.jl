# `gaussian_approximation` IFT path for `WorkspaceGMRF` priors when EITHER the
# prior carries `ForwardDiff.Dual` partials (μ / Q derived from outer-AD
# hyperparameters) OR the observation likelihood does (Dual hyperparams in
# `AutoDiffLikelihood` or any component of `CompositeLikelihood`).
#
# Unified sequence (pure AD — no finite differences):
#   1. Strip Duals from prior + lik → primal Newton on the stripped pair.
#   2. At the converged primal x*, compute ∂(neg_grad)/∂θ via exact AD.
#      Lift x* to `Dual{outer_tag, Float64, N}` with zero outer-partials and
#      evaluate `Q_prior·(x_dual - μ_prior) - loggrad(x_dual, lik)` against
#      the ORIGINAL (Dual-carrying) prior + lik. The primal residual is
#      zero in the unconstrained case (Newton condition); for the
#      constrained case its tangent-space projection is zero and the
#      projection step at IFT solve time uses ci.A_tilde_T / ci.L_c. The
#      outer partials of the result are ∂(neg_grad)/∂θ at fixed x*. For a
#      Dual prior they include `∂Q_prior/∂θ · (x* - μ_prior) - Q_prior ·
#      ∂μ_prior/∂θ`; for a Dual lik they include `-∂loggrad/∂θ`. Either
#      subset works without special-casing.
#   3. Solve Q_post · dx*/dθ_j = -∂(neg_grad)/∂θ_j with the primal posterior
#      factor. Project onto the constraint tangent if constrained.
#   4. Build x_star_dual carrying dx/dθ as outer partials.
#   5. Compute the total `dH/dθ` via one `loghessian` call on `x_star_dual`
#      against the Dual lik. The result captures both the explicit
#      θ-derivative and the implicit x*(θ)-derivative.
#   6. Assemble Dual `Q_post` by subtracting H from a `DualT`-lifted copy of
#      Q_prior's nzval, preserving Q_prior's exact sparse pattern.

"""
    _is_dual_autodifflik(lik) -> Bool

True when `lik::AutoDiffLikelihood` and any of its stored hyperparams
carries Dual partials.
"""
_is_dual_autodifflik(lik) = false
function _is_dual_autodifflik(lik::GMRFs.AutoDiffLikelihood)
    return GMRFs._hp_carries_ad_partials(lik.hyperparams)
end

"""
    _lik_carries_dual_hp(lik) -> Bool

True if `lik` (or any component of a `CompositeLikelihood`) carries Dual-
valued hyperparameters that the IFT path needs to thread through.
Recognises `AutoDiffLikelihood` (via stored hyperparams), the per-channel
`_DualObsLik` types (Normal / NegBin / Gamma / StudentT with Dual fields),
and `CompositeLikelihood` (recurses).
"""
_lik_carries_dual_hp(lik) = false
_lik_carries_dual_hp(lik::GMRFs.AutoDiffLikelihood) = _is_dual_autodifflik(lik)
_lik_carries_dual_hp(::_DualObsLik) = true
_lik_carries_dual_hp(lik::GMRFs.CompositeLikelihood) =
    any(_lik_carries_dual_hp, lik.components)
_lik_carries_dual_hp(lik::GMRFs.LinearlyTransformedLikelihood) =
    _lik_carries_dual_hp(lik.base_likelihood) || GMRFs._carries_ad_partials(lik.design_matrix)
_lik_carries_dual_hp(lik::GMRFs.NonlinearLeastSquaresLikelihood) =
    GMRFs._hp_carries_ad_partials(lik.hyperparams) || GMRFs._carries_ad_partials(lik.inv_σ²)

# ----------------------------------------------------------------------------
# DI.hessian returns a dense Matrix even when the underlying Hessian is
# structurally diagonal (the typical case for a sum-of-pointwise loglik
# where each term `i` depends only on `x[i]`). Detect off-diagonal zeros
# and downcast so the IFT path can preserve Q_prior's sparse pattern
# through `_assemble_q_post_dual`. With both `pointwise_loglik_func` and
# `diagonal_hessian_safe = true` set, `loghessian` already returns a
# `Diagonal` directly via the main-src fast path, so this is a defensive
# fallback.
# ----------------------------------------------------------------------------

_maybe_downcast_diagonal(H::Diagonal) = H
_maybe_downcast_diagonal(H::SparseMatrixCSC) = H
# `LinearlyTransformedLikelihood.loghessian` returns `Symmetric(A' * hess_η * A)`,
# whose underlying matrix is fully populated by the multiplication. Unwrap so the
# downstream sparse-vs-dense dispatch sees the real storage type.
_maybe_downcast_diagonal(H::Symmetric) = _maybe_downcast_diagonal(parent(H))
function _maybe_downcast_diagonal(H::AbstractMatrix)
    n = size(H, 1)
    n == size(H, 2) || return H
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        iszero(H[i, j]) || return H
    end
    return Diagonal([H[i, i] for i in 1:n])
end

# ----------------------------------------------------------------------------
# Q_post_dual builder — preserves Q_prior's exact sparse structure for
# diagonal Hessians (the dominant case when `pointwise_loglik_func` is set
# AND `diagonal_hessian_safe = true`). For subset-pattern sparse Hessians
# we still preserve Q_prior's pattern. For arbitrary dense Hessians we
# fall back to algebraic subtraction.
# ----------------------------------------------------------------------------

# Allocate `nzval_dual` as a copy of Q_prior.nzval typed as `DualT`. For a
# Float64 Q_prior we lift each entry with zero partials; for a Dual Q_prior
# (whose Tag/N must already match `DualT`) we just copy. Both sparse-
# pattern-preserving variants below start from this and subtract H entries
# in-place wherever they intersect Q_prior's pattern.
function _alloc_dual_nzval_from_qprior(
        Q_prior::SparseMatrixCSC{Float64}, ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))
    nzval_dual = Vector{DualT}(undef, length(Q_prior.nzval))
    @inbounds for i in eachindex(Q_prior.nzval)
        nzval_dual[i] = DualT(Q_prior.nzval[i], zero_partials)
    end
    return nzval_dual
end

function _alloc_dual_nzval_from_qprior(
        Q_prior::SparseMatrixCSC{DualT}, ::Type{DualT}, ::Val{N}
    ) where {DualT <: ForwardDiff.Dual, N}
    return copy(Q_prior.nzval)
end

_q_post_with_pattern(Q_prior::SparseMatrixCSC, nzval_dual) =
    SparseMatrixCSC(Q_prior.m, Q_prior.n, copy(Q_prior.colptr), copy(Q_prior.rowval), nzval_dual)

# Diagonal H_dual — subtract H from a `DualT`-lifted copy of Q_prior's nzval.
# Works for both Float64 and Dual Q_prior; native `Dual - Dual` arithmetic
# composes the prior's existing partials with H's.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC, H_dual::Diagonal{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    n = size(Q_prior, 1)
    nzval_dual = _alloc_dual_nzval_from_qprior(Q_prior, DualT, Val(N))
    @inbounds for j in 1:n
        for k in nzrange(Q_prior, j)
            if Q_prior.rowval[k] == j
                nzval_dual[k] -= H_dual.diag[j]
                break
            end
        end
    end
    return _q_post_with_pattern(Q_prior, nzval_dual)
end

# Sparse non-diagonal H_dual — match Q_prior's pattern, subtract H values
# where they overlap. Requires H's pattern to be a subset of Q_prior's;
# otherwise we'd silently drop H nonzeros outside Q_prior and return a wrong
# Q_post. Errors loudly in that case so the caller knows workspace reuse
# isn't applicable for this likelihood/prior combination.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC,
        H_dual::SparseMatrixCSC{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    _check_h_pattern_subset(H_dual, Q_prior)
    n = size(Q_prior, 1)
    nzval_dual = _alloc_dual_nzval_from_qprior(Q_prior, DualT, Val(N))
    @inbounds for col in 1:n
        for k in nzrange(Q_prior, col)
            row = Q_prior.rowval[k]
            h = _sparse_lookup(H_dual, row, col)
            nzval_dual[k] -= h
        end
    end
    return _q_post_with_pattern(Q_prior, nzval_dual)
end

# Verify every nonzero of H is at a (row, col) that's also nonzero in
# Q_prior. If not, the IFT-with-workspace path can't preserve Q_prior's
# sparse pattern, and silently returning a Q_post built only from
# Q_prior's pattern would drop real H entries.
function _check_h_pattern_subset(H::SparseMatrixCSC, Q_prior::SparseMatrixCSC)
    for col in 1:size(H, 2)
        for k in nzrange(H, col)
            iszero(H.nzval[k]) && continue
            row = H.rowval[k]
            _sparse_lookup_present(Q_prior, row, col) || throw(
                ArgumentError(
                    "AutoDiffLikelihood IFT path: observation Hessian has a nonzero at " *
                        "($row, $col) outside the prior precision sparsity pattern. The " *
                        "workspace-reuse path requires H's pattern to be a subset of Q_prior. " *
                        "Use a likelihood whose Hessian is structurally diagonal " *
                        "(supply `pointwise_loglik_func` AND set `diagonal_hessian_safe = true`) " *
                        "or call `gaussian_approximation` without a `WorkspaceGMRF` prior."
                )
            )
        end
    end
    return nothing
end

function _sparse_lookup_present(A::SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in nzrange(A, col)
        A.rowval[k] == row && return true
    end
    return false
end

# Generic / dense H_dual — error path. The workspace IFT pipeline produces
# a `WorkspaceGMRF`, whose constructor demands `SparseMatrixCSC` precision
# matching the workspace's symbolic factorization pattern. Returning a
# dense `Matrix` here would just shift the failure to the constructor with
# a less actionable message, so we error explicitly with guidance.
function _assemble_q_post_dual(
        Q_prior::SparseMatrixCSC,
        H_dual::AbstractMatrix{<:ForwardDiff.Dual},
        ::Type{DualT}, ::Val{N}
    ) where {DualT, N}
    throw(
        ArgumentError(
            "AutoDiffLikelihood IFT path: observation Hessian is dense " *
                "(eltype $(eltype(H_dual)), size $(size(H_dual))), but the " *
                "workspace-reuse path requires a structurally diagonal or " *
                "Q_prior-pattern-subset sparse Hessian to preserve the workspace's " *
                "symbolic factorization. Supply `pointwise_loglik_func` AND set " *
                "`diagonal_hessian_safe = true` to opt into the structured Diagonal " *
                "path, use a sparse Hessian backend (`AutoSparse(AutoForwardDiff())`), " *
                "or call `gaussian_approximation` with a non-`WorkspaceGMRF` prior."
        )
    )
end

function _sparse_lookup(A::SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in nzrange(A, col)
        A.rowval[k] == row && return A.nzval[k]
    end
    return zero(eltype(A))
end

# ----------------------------------------------------------------------------
# Outer-tag extraction across prior + lik (incl. composite components).
#
# All Dual partials threaded through the IFT must originate from a single
# outer-AD pass — same Tag, same chunk size N. The collectors below traverse
# the prior and the (possibly composite) likelihood, returning the unique
# (Tag, N) or `(nothing, nothing)` for purely Float64 inputs. Mismatched
# tags / chunk sizes error loudly because misreading partials silently in
# steps 2/5 would give wrong gradients.
# ----------------------------------------------------------------------------

_lik_dual_tag_npartials(::Any) = (nothing, nothing)
_lik_dual_tag_npartials(lik::GMRFs.AutoDiffLikelihood) =
    _is_dual_autodifflik(lik) ? _outer_tag_and_npartials(lik.hyperparams) : (nothing, nothing)

function _lik_dual_tag_npartials(lik::_DualObsLik)
    D = _dual_type_from_obs_lik(lik)
    return ForwardDiff.tagtype(D), ForwardDiff.npartials(D)
end

function _lik_dual_tag_npartials(lik::GMRFs.LinearlyTransformedLikelihood)
    T1, N1 = _lik_dual_tag_npartials(lik.base_likelihood)
    T2, N2 = _array_tag_npartials(lik.design_matrix)
    return _reconcile_tag_npartials(T1, N1, T2, N2)
end

function _lik_dual_tag_npartials(lik::GMRFs.NonlinearLeastSquaresLikelihood)
    T1, N1 = GMRFs._hp_carries_ad_partials(lik.hyperparams) ?
        _outer_tag_and_npartials(lik.hyperparams) : (nothing, nothing)
    T2, N2 = _array_tag_npartials(lik.inv_σ²)
    return _reconcile_tag_npartials(T1, N1, T2, N2)
end

# (Tag, N) of a Dual-eltype array, or (nothing, nothing) for a plain array.
_array_tag_npartials(A::AbstractArray{<:ForwardDiff.Dual}) =
    (ForwardDiff.tagtype(eltype(A)), ForwardDiff.npartials(eltype(A)))
_array_tag_npartials(::AbstractArray) = (nothing, nothing)

# Combine two (Tag, N) results, erroring if both are present but disagree.
function _reconcile_tag_npartials(T1, N1, T2, N2)
    T1 === nothing && return (T2, N2)
    T2 === nothing && return (T1, N1)
    (T1 === T2 && N1 == N2) || throw(
        ArgumentError(
            "Observation likelihood carries Duals from different outer-AD passes " *
                "(Tag=$T1/N=$N1 vs Tag=$T2/N=$N2). All Dual partials threaded through " *
                "the IFT must come from a single outer ForwardDiff pass."
        )
    )
    return (T1, N1)
end

function _lik_dual_tag_npartials(lik::GMRFs.CompositeLikelihood)
    Tag, N = nothing, nothing
    for (i, comp) in enumerate(lik.components)
        T_c, N_c = _lik_dual_tag_npartials(comp)
        T_c === nothing && continue
        if Tag === nothing
            Tag, N = T_c, N_c
        elseif T_c !== Tag || N_c != N
            throw(
                ArgumentError(
                    "CompositeLikelihood components carry Duals from different " *
                        "outer-AD passes (component $i has Tag=$T_c / N=$N_c, " *
                        "expected Tag=$Tag / N=$N). All Dual hyperparams must come " *
                        "from a single outer ForwardDiff pass."
                )
            )
        end
    end
    return Tag, N
end

_prior_dual_tag_npartials(::GMRFs.WorkspaceGMRF{Float64}) = (nothing, nothing)
function _prior_dual_tag_npartials(prior::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    D = eltype(prior.mean)
    return ForwardDiff.tagtype(D), ForwardDiff.npartials(D)
end

function _ift_outer_tag_and_npartials(prior, lik)
    Tag, N = _prior_dual_tag_npartials(prior)
    Tag2, N2 = _lik_dual_tag_npartials(lik)
    if Tag2 !== nothing
        if Tag === nothing
            Tag, N = Tag2, N2
        elseif Tag !== Tag2 || N != N2
            throw(
                ArgumentError(
                    "WorkspaceGMRF prior and observation likelihood carry Duals " *
                        "from different outer-AD passes (prior: Tag=$Tag / N=$N, " *
                        "lik: Tag=$Tag2 / N=$N2). All Dual partials threaded " *
                        "through the IFT must come from a single outer ForwardDiff pass."
                )
            )
        end
    end
    Tag === nothing &&
        throw(ArgumentError("IFT path invoked with no Dual partials in either prior or likelihood"))
    return Tag, N
end

# ----------------------------------------------------------------------------
# Primal stripping: build a Float64 prior + Float64 lik for the inner Newton.
# `_primal_obs_lik` is extended (in autodiff_likelihood_dual.jl) to handle
# `AutoDiffLikelihood` and `CompositeLikelihood` recursively.
# ----------------------------------------------------------------------------

_primal_workspace_gmrf_for_ift(p::GMRFs.WorkspaceGMRF{Float64}) = p
function _primal_workspace_gmrf_for_ift(p::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    return p.constraints === nothing ?
        _primal_workspace_gmrf(p) :
        _primal_constrained_workspace_gmrf(p)
end

# ----------------------------------------------------------------------------
# Core IFT helper: handles every combination of Float64/Dual prior with
# AutoDiffLikelihood / CompositeLikelihood / _DualObsLik (and any future
# obs_lik whose `loggrad`/`loghessian` correctly propagate Dual hp).
# ----------------------------------------------------------------------------

# Residual-curvature correction for the IFT mode-sensitivity Hessian. `nothing` for
# likelihoods whose posterior precision already equals the true Hessian; a sparse `C`
# for Gauss–Newton least squares (so the true Hessian is `Q_post - C`).
_ift_hessian_correction(::GMRFs.ObservationLikelihood, primal_lik, x_star) = nothing
_ift_hessian_correction(::GMRFs.NonlinearLeastSquaresLikelihood, primal_lik, x_star) =
    GMRFs.residual_curvature(primal_lik, x_star)

# Solver for the mode-sensitivity system. Without a correction, reuse the workspace's
# Gauss–Newton factorization. With a correction `C`, the true Hessian `Q_post - C` is
# factorized once (it is positive definite at the mode) and reused for every RHS.
_ift_sensitivity_solver(ws::GMRFs.GMRFWorkspace, ::Nothing) = rhs -> GMRFs.workspace_solve(ws, rhs)
function _ift_sensitivity_solver(ws::GMRFs.GMRFWorkspace, C::AbstractMatrix)
    H_true = GMRFs._snapshot_Q(ws) - C
    F = cholesky(Symmetric(H_true))
    return rhs -> (F \ rhs)
end

function _workspace_dualhp_ift(
        prior_gmrf::GMRFs.WorkspaceGMRF,
        obs_lik::GMRFs.ObservationLikelihood;
        ga_kwargs...
    )
    OuterTag, N = _ift_outer_tag_and_npartials(prior_gmrf, obs_lik)
    DualT = ForwardDiff.Dual{OuterTag, Float64, N}
    PartialsT = ForwardDiff.Partials{N, Float64}
    zero_partials = PartialsT(ntuple(_ -> 0.0, Val(N)))

    # Step 1: primal Newton on stripped prior + stripped lik.
    primal_prior = _primal_workspace_gmrf_for_ift(prior_gmrf)
    primal_lik = _primal_obs_lik(obs_lik)
    posterior_primal = GMRFs.gaussian_approximation(primal_prior, primal_lik; ga_kwargs...)
    x_star = GMRFs.mean(posterior_primal)
    n = length(x_star)

    # Step 2: ∂(neg_grad)/∂θ at fixed x* via exact AD.
    #
    # Lift x* to DualT with zero outer partials. With Dual prior and/or Dual
    # lik in scope, evaluating
    #
    #     neg_grad = Q_prior · (x_dual - μ_prior) - loggrad(x_dual, lik)
    #
    # gives a Dual whose primal value is zero (Newton condition) and whose
    # outer partials encode ∂(neg_grad)/∂θ_j. The IFT relation
    # `Q_post · dx*/dθ_j = -∂(neg_grad)/∂θ_j` then yields the tangent solves
    # uniformly across all four prior×lik Dual/Float64 subcases.
    x_star_dual_zero = [DualT(x_star[i], zero_partials) for i in 1:n]
    neg_grad_dual = prior_gmrf.precision * (x_star_dual_zero .- prior_gmrf.mean) .-
        GMRFs.loggrad(x_star_dual_zero, obs_lik)
    rhs_dθ = ntuple(
        j -> [-ForwardDiff.partials(neg_grad_dual[i], j) for i in 1:n],
        Val(N)
    )

    # Step 3: IFT solves on the primal posterior workspace.
    #
    # The mode sensitivity solves `H · dx*/dθ = -∂(neg_grad)/∂θ`, where `H` is the
    # TRUE Hessian of the neg-log-posterior at `x*`. For most likelihoods the
    # posterior precision equals that Hessian, so the workspace's factorization is
    # reused directly. Gauss–Newton likelihoods (NLSQ) use `JᵀWJ` for the posterior
    # precision, which differs from the true Hessian by the residual-curvature term
    # `C`; `_ift_sensitivity_solver` then factorizes the corrected `Q_post - C`.
    ws = posterior_primal.workspace
    constrained = posterior_primal.constraints !== nothing
    ci = constrained ? posterior_primal.constraints : nothing
    solve_dθ = _ift_sensitivity_solver(ws, _ift_hessian_correction(obs_lik, primal_lik, x_star))
    dx = Matrix{Float64}(undef, n, N)
    for j in 1:N
        step = solve_dθ(rhs_dθ[j])
        if constrained
            A = ci.matrix
            step = step - ci.A_tilde_T * (ci.L_c \ (A * step))
        end
        dx[:, j] .= step
    end

    # Step 4: assemble Dual x* with dx/dθ as outer partials.
    x_star_dual = [
        DualT(x_star[i], PartialsT(ntuple(j -> dx[i, j], Val(N))))
            for i in 1:n
    ]

    # Step 5: total dH/dθ via one exact-AD loghessian call against the
    # original (Dual-carrying) lik. Captures both ∂H/∂θ and ∂H/∂x · dx/dθ.
    H_dual = _maybe_downcast_diagonal(GMRFs.loghessian(x_star_dual, obs_lik))
    Q_post_dual = _assemble_q_post_dual(prior_gmrf.precision, H_dual, DualT, Val(N))

    # Step 6: result, preserving constraints if the prior had them. Reuses
    # the primal posterior's already-computed Float64 Ã^T / L_c / log_AA_det
    # for the constrained branch — the Dual-mean / Dual-Q corrections are
    # rebuilt inside `_build_constrained_dual_workspace_gmrf`.
    if prior_gmrf.constraints === nothing
        return GMRFs.WorkspaceGMRF(x_star_dual, Q_post_dual, ws)
    else
        ci_post = posterior_primal.constraints
        A_dense = ci_post.matrix
        log_AA_det = logdet(cholesky(Symmetric(A_dense * A_dense')))
        return _build_constrained_dual_workspace_gmrf(
            x_star_dual, Q_post_dual, ws,
            A_dense, ci_post.vector, ci_post.A_tilde_T, ci_post.L_c,
            log_AA_det, posterior_primal.version
        )
    end
end

# ----------------------------------------------------------------------------
# Dispatch hooks: route Dual-bearing prior×lik combinations through the IFT.
# A single `Union` over the two lik families collapses what would otherwise
# be four near-identical methods. Kept *separate* from the older
# `WorkspaceGMRF{D<:Dual} + ObservationLikelihood` dispatch in
# `workspace_gaussian_approximation.jl` because Julia's resolution would
# pick the broader method for `_DualObsLik` types we don't want to
# redirect through the unified path.
# ----------------------------------------------------------------------------

const _WorkspaceDualHpIFTLik = Union{
    GMRFs.AutoDiffLikelihood, GMRFs.CompositeLikelihood,
    GMRFs.LinearlyTransformedLikelihood, GMRFs.NonlinearLeastSquaresLikelihood,
}

# Float64 prior — IFT only when the lik (or one of its components) carries
# Duals. Otherwise fall through to the primal Newton path.
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{Float64},
        obs_lik::_WorkspaceDualHpIFTLik;
        kwargs...
    )
    _lik_carries_dual_hp(obs_lik) &&
        return _workspace_dualhp_ift(prior_gmrf, obs_lik; kwargs...)
    return invoke(
        GMRFs.gaussian_approximation,
        Tuple{GMRFs.WorkspaceGMRF, GMRFs.ObservationLikelihood},
        prior_gmrf, obs_lik;
        kwargs...
    )
end

# Dual prior — always IFT. The prior's Duals would otherwise be stripped
# away by the primal Newton's Float64 workspace nzval.
function GMRFs.gaussian_approximation(
        prior_gmrf::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual},
        obs_lik::_WorkspaceDualHpIFTLik;
        kwargs...
    )
    return _workspace_dualhp_ift(prior_gmrf, obs_lik; kwargs...)
end
