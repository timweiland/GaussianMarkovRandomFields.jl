# Dual-valued `var` for `GMRF`, `WorkspaceGMRF` and `ConstrainedGMRF`.
#
# These types deliberately hold a *primal* factorization — CHOLMOD and friends
# cannot store Duals — so the Float64 path's `selinv_diag` hands back plain
# `Float64`s and every partial is dropped. Without the methods below `var`
# reports a zero gradient rather than failing.
#
# `logdetcov` works around the primal cache with an IFT correction (see
# `logdetcov.jl`), but `var` has no such shortcut: `d(diag(Q⁻¹)) = -diag(Q⁻¹ Q̇ Q⁻¹)`
# reaches entries of `Q⁻¹` outside the selected-inverse pattern. Redoing the
# selected inversion in Dual arithmetic *is* cheap, though — the Takahashi
# recursion is closed over the fill-in pattern, so its tangent stays in-pattern
# too. `ChordalCholesky` is generic Julia and propagates partials exactly, which
# is why `ChordalGMRF` never had this bug; these methods route `GMRF` through the
# same machinery. The cost is a fresh Dual factorization per call, since the
# cached primal one cannot be reused.

function var(d::GMRF{<:ForwardDiff.Dual})
    return _dual_var_impl(d, GMRFs.supports_selinv(d.linsolve_cache.alg))
end

function _dual_var_impl(d::GMRF{<:ForwardDiff.Dual}, ::Val{true})
    return _dual_selinv_diag(GMRFs.precision_matrix(d))
end

# The Float64 fallback for solvers without selected inversion is `RBMCStrategy`,
# a Monte Carlo estimator whose solves go through the primal cache. Its partials
# would be dropped exactly like the ones this file exists to fix, so refuse
# rather than hand back another silent zero.
function _dual_var_impl(d::GMRF{<:ForwardDiff.Dual}, ::Val{false})
    throw(
        ArgumentError(
            "Cannot differentiate `var(::GMRF)` with algorithm $(typeof(d.linsolve_cache.alg)): " *
                "it does not support selected inversion, so `var` falls back to the stochastic " *
                "`RBMCStrategy` estimator, which cannot propagate ForwardDiff partials. " *
                "Construct the GMRF with a selected-inversion-capable solver — e.g. " *
                "`GMRF(mean, precision, LinearSolve.CHOLMODFactorization())` — to differentiate `var`."
        )
    )
end

# `WorkspaceGMRF` keeps the same split — Dual `precision` field, primal workspace —
# so `selinv_diag(d.workspace)` drops partials exactly like the `GMRF` case.
function var(d::GMRFs.WorkspaceGMRF{<:ForwardDiff.Dual})
    if d.constraints !== nothing
        throw(
            ArgumentError(
                "Cannot differentiate `var` through a constrained `WorkspaceGMRF`: the " *
                    "stored `ConstraintInfo` is built from the primal factorization, so the " *
                    "constraint correction would contribute no partials."
            )
        )
    end
    return _dual_selinv_diag(d.precision)
end

# `var(::ConstrainedGMRF)` subtracts a correction built from the *stored* `L_c`
# and `A_tilde_T`, both Float64. Differentiating only the base term would give a
# gradient that looks reasonable but is simply the unconstrained one, so rebuild
# the constraint factors as Duals first.
function var(d::GMRFs.ConstrainedGMRF{<:ForwardDiff.Dual})
    return _dual_constrained_var(d, d.base_gmrf)
end

function _dual_constrained_var(
        d::GMRFs.ConstrainedGMRF, base_gmrf::GMRF{<:ForwardDiff.Dual}
    )
    σ_base = var(base_gmrf)
    A_tilde_T_dual, L_c_dual =
        _dual_constraint_factors(base_gmrf, d.constraint_matrix, d.A_tilde_T)

    # σ_c = σ - diag(Ã^T (AÃ^T)⁻¹ Ã) = σ - colsums((L_c.L \ Ã^Tᵀ).^2)
    B_T = L_c_dual.L \ A_tilde_T_dual'
    σ = σ_base .- vec(sum(abs2, B_T, dims = 1))
    # Match the Float64 path, which clamps away tiny negative round-off.
    return max.(σ, zero(eltype(σ)))
end

function _dual_constrained_var(d::GMRFs.ConstrainedGMRF, base_gmrf)
    throw(
        ArgumentError(
            "Cannot differentiate `var(::ConstrainedGMRF)` with a base GMRF of type " *
                "$(typeof(base_gmrf)): rebuilding the constraint factors under Dual " *
                "arithmetic is only implemented for a `GMRF` base."
        )
    )
end

# Diagonal precision inverts elementwise; no factorization needed.
_dual_selinv_diag(Q::Diagonal) = inv.(Q.diag)

function _dual_selinv_diag(Q::AbstractMatrix)
    H = Hermitian(SparseArrays.sparse(Q), :L)
    Σ = mselinv(H, cholesky!(ChordalCholesky(H)))
    # `diag` of a sparse matrix is a `SparseVector`; the Float64 path returns a
    # dense `Vector`, and `ForwardDiff.jacobian` cannot extract partials from a
    # sparse result. Marginal variances are dense anyway.
    return Vector(diag(Σ))
end
