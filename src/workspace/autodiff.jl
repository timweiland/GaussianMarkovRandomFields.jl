using ChainRulesCore
using Distributions: logpdf
using LinearAlgebra
using SparseArrays

# --- logpdf rrule for WorkspaceGMRF (handles both constrained and unconstrained) ---

function ChainRulesCore.rrule(::typeof(logpdf), x::WorkspaceGMRF, z::AbstractVector)
    μ_base = x.mean  # unconstrained mean (for precision gradient computation)
    Q = precision_matrix(x)
    r = z - μ_base
    val = logpdf(x, z)

    function workspace_logpdf_pullback(ȳ)
        ensure_loaded!(x)
        Qinv = selinv(x.workspace)
        Qr = Q * r

        μ̄ = ȳ * Qr
        Q̄ = compute_precision_gradient(Qinv, r, ȳ)

        # Constraint correction contributions (Rue & Held 2005, §2.3.3).
        # Math mirrors the ConstrainedGMRF rrule in src/autodiff/constructors.jl.
        if has_constraints(x)
            ci = x.constraints
            A = ci.matrix
            resid_e = ci.vector - A * x.mean

            # μ̄ contribution: -ȳ * A' * (L_c \ resid_e)
            μ̄ = collect(μ̄) .- ȳ .* (A' * (ci.L_c \ resid_e))

            # Q̄ contribution: -0.5 * ȳ * A_tilde_T * (S⁻¹ - w*w') * A_tilde_T'
            S_inv = inv(ci.L_c)
            w = S_inv * resid_e
            Q̄_corr = (-0.5 * ȳ) .* (ci.A_tilde_T * (S_inv - w * w') * ci.A_tilde_T')
            Q̄ = Q̄ + Q̄_corr
        end

        x̄ = Tangent{typeof(x)}(;
            mean = μ̄,
            precision = Q̄,
            workspace = NoTangent(),
            constraints = NoTangent(),
            version = NoTangent()
        )

        z̄ = ȳ * (-Qr)

        return NoTangent(), x̄, z̄
    end

    return val, workspace_logpdf_pullback
end

# --- logdetcov rrule for WorkspaceGMRF ---

# ∂logdetcov/∂Q = -Q⁻¹ on Q's own sparsity pattern; see the commentary in
# src/autodiff/logdetcov.jl for why the selected inverse is exact here.
#
# Constraints do not appear: `logdetcov(::WorkspaceGMRF)` is the *base*
# log-determinant, and the Rue-Held correction is a separate term that `logpdf`
# adds on top of it. So this rule is the same whether or not `x` is constrained,
# and it must NOT pick up a constraint contribution — the regression test pins
# that by checking a constrained workspace against the unconstrained
# -logdet(Q) reference.
function ChainRulesCore.rrule(::typeof(logdetcov), x::WorkspaceGMRF)
    val = logdetcov(x)

    function workspace_logdetcov_pullback(ȳ)
        c = -unthunk(ȳ)
        # Skip the selected inversion entirely when the result is unused.
        c isa AbstractZero && return NoTangent(), ZeroTangent()

        # The workspace is shared and may have been refactorized at a different
        # Q since the forward pass, so re-load before reading the selected
        # inverse (matching the logpdf pullback above).
        ensure_loaded!(x)
        Q̄ = c * selinv(x.workspace)

        x̄ = Tangent{typeof(x)}(;
            mean = ZeroTangent(),
            precision = Q̄,
            workspace = NoTangent(),
            constraints = NoTangent(),
            version = NoTangent(),
        )
        return NoTangent(), x̄
    end

    return val, workspace_logdetcov_pullback
end

# --- WorkspaceGMRF constructor rrules ---

function ChainRulesCore.rrule(
        ::Type{WorkspaceGMRF}, μ::AbstractVector, Q::SparseMatrixCSC
    )
    x = WorkspaceGMRF(μ, Q)

    function WorkspaceGMRF_pullback(x̄)
        return NoTangent(), x̄.mean, x̄.precision
    end

    return x, WorkspaceGMRF_pullback
end

function ChainRulesCore.rrule(
        ::Type{WorkspaceGMRF}, μ::AbstractVector, Q::SparseMatrixCSC, ws::GMRFWorkspace
    )
    x = WorkspaceGMRF(μ, Q, ws)

    function WorkspaceGMRF_ws_pullback(x̄)
        return NoTangent(), x̄.mean, x̄.precision, NoTangent()
    end

    return x, WorkspaceGMRF_ws_pullback
end

# Constrained constructor rrule. The ConstraintInfo (constrained_mean, L_c, etc.)
# is treated as derived from (μ, Q) via the symbolic factorization — gradients
# through it are handled by the logpdf rrule below, which differentiates through
# log_constraint_correction directly. Suitable for pipelines that consume the
# constrained WorkspaceGMRF via logpdf or rand; do NOT rely on this rrule if
# you also differentiate through `mean(d)` (which returns the constrained mean).
function ChainRulesCore.rrule(
        ::Type{WorkspaceGMRF}, μ::AbstractVector, Q::SparseMatrixCSC,
        ws::GMRFWorkspace, A::AbstractMatrix, e::AbstractVector
    )
    x = WorkspaceGMRF(μ, Q, ws, A, e)

    function WorkspaceGMRF_constrained_pullback(x̄)
        return NoTangent(), x̄.mean, x̄.precision, NoTangent(), NoTangent(), NoTangent()
    end

    return x, WorkspaceGMRF_constrained_pullback
end

# --- gaussian_approximation rrule for WorkspaceGMRF ---

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(gaussian_approximation),
        prior_gmrf::WorkspaceGMRF,
        obs_lik::ObservationLikelihood;
        kwargs...
    )
    # Reverse-mode AD can't differentiate the Gauss–Newton sparse Jacobian; fail with
    # actionable guidance rather than deep in AD internals. (Forward-mode is exact.)
    _has_gauss_newton_jacobian(obs_lik) && _reverse_mode_gauss_newton_error()

    posterior = gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
    x_star = mean(posterior)

    function workspace_ga_pullback(ȳ)
        μ̄ = ȳ.mean
        Q̄ = ȳ.precision

        if _is_zero_tangent(Q̄)
            x_tangent_from_hess = nothing
            obs_lik_tangent_from_Q̄ = NoTangent()
        else
            _, hess_pullback = rrule_via_ad(config, loghessian, x_star, obs_lik)
            _, x_tangent_from_hess, obs_lik_tangent_from_Q̄ = hess_pullback(-Q̄)
            # rrule_via_ad may hand back Thunks (e.g. from ChainRules' sparse
            # A*x rule when obs_lik carries a design matrix); `collect` and
            # `_add_namedtuples` below need materialized tangents.
            x_tangent_from_hess = unthunk(x_tangent_from_hess)
            obs_lik_tangent_from_Q̄ = unthunk(obs_lik_tangent_from_Q̄)
        end

        ensure_loaded!(posterior)
        ws = posterior.workspace
        b_ift = _is_zero_tangent(x_tangent_from_hess) ?
            collect(μ̄) :
            collect(μ̄) .+ collect(x_tangent_from_hess)
        λ = workspace_solve(ws, b_ift)

        # For the VJP, use the unconstrained base prior
        base_prior = has_constraints(prior_gmrf) ?
            WorkspaceGMRF(prior_gmrf.mean, prior_gmrf.precision, prior_gmrf.workspace) :
            prior_gmrf

        _, ∇_pullback = rrule_via_ad(
            config, ∇ₓ_neg_log_posterior, base_prior, obs_lik, x_star
        )
        _, prior_tangent, obs_lik_tangent, _ = ∇_pullback(-λ)
        # A Thunk here would silently fail the `isa Tangent` checks downstream
        # and drop gradient terms, so materialize defensively.
        prior_tangent = unthunk(prior_tangent)
        obs_lik_tangent = unthunk(obs_lik_tangent)

        if !_is_zero_tangent(Q̄)
            prior_tangent = _workspace_add_precision_tangent(prior_tangent, prior_gmrf, Q̄)
        end

        obs_lik_combined = _add_namedtuples(obs_lik_tangent, obs_lik_tangent_from_Q̄)

        return (NoTangent(), prior_tangent, obs_lik_combined)
    end

    return posterior, workspace_ga_pullback
end

# --- rrule shield for the LatentModel-on-workspace callable ---

# Positional θ-wrappers for rrule_via_ad: ChainRules treats kwargs as
# non-differentiable, so the hyperparameters are re-exposed as a NamedTuple
# positional argument that reverse-mode AD can produce tangents for.
_model_mean_from_θ(model::LatentModel, θ::NamedTuple) = mean(model; θ...)
_model_precision_from_θ(model::LatentModel, θ::NamedTuple) =
    _ensure_sparse(precision_matrix(model; θ...))

# Accumulate two tangents where either may be a zero-like tangent.
_accum_tangent(a, b) = _is_zero_tangent(a) ? b : (_is_zero_tangent(b) ? a : a + b)

"""
    ChainRulesCore.rrule(config, ::typeof(_evaluate_with_workspace), model, ws, θ)

Shield the `(model::LatentModel)(ws::GMRFWorkspace; θ...)` fast path from
reverse-mode AD: the primal runs as-is (including the `update_precision!`
mutation, which Zygote cannot trace), and the pullback routes the
`WorkspaceGMRF` tangent's `mean`/`precision` components through
`rrule_via_ad` of `mean(model; θ...)` / `precision_matrix(model; θ...)`.

The workspace itself is non-differentiable state. Constraint quantities
(`ȳ.constraints`) are ignored, matching the constrained `WorkspaceGMRF`
constructor rrule above: the constrained-logpdf correction terms already
arrive folded into `ȳ.mean`/`ȳ.precision`.

`Q̄` may live on the workspace's (possibly obs-padded) sparsity pattern; the
padded positions hold structural zeros of the model's precision, so the
pullback of the model's own construction correctly ignores them.
"""
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(_evaluate_with_workspace),
        model::LatentModel, ws::GMRFWorkspace, θ::NamedTuple
    )
    gmrf = _evaluate_with_workspace(model, ws, θ)

    function _evaluate_with_workspace_pullback(ȳ)
        ȳ = unthunk(ȳ)
        if ȳ isa AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        μ̄ = unthunk(ȳ.mean)
        Q̄ = unthunk(ȳ.precision)

        model_tangent = NoTangent()
        θ̄ = NoTangent()

        if !_is_zero_tangent(μ̄)
            _, mean_pb = rrule_via_ad(config, _model_mean_from_θ, model, θ)
            _, m̄_mean, θ̄_mean = mean_pb(collect(μ̄))
            model_tangent = _accum_tangent(model_tangent, unthunk(m̄_mean))
            θ̄ = _accum_tangent(θ̄, unthunk(θ̄_mean))
        end

        if !_is_zero_tangent(Q̄)
            _, prec_pb = rrule_via_ad(config, _model_precision_from_θ, model, θ)
            _, m̄_prec, θ̄_prec = prec_pb(Q̄)
            model_tangent = _accum_tangent(model_tangent, unthunk(m̄_prec))
            θ̄ = _accum_tangent(θ̄, unthunk(θ̄_prec))
        end

        return NoTangent(), model_tangent, NoTangent(), θ̄
    end

    return gmrf, _evaluate_with_workspace_pullback
end

# --- Helper: add Q̄ to prior tangent for WorkspaceGMRF ---

function _workspace_add_precision_tangent(prior_tangent, prior::WorkspaceGMRF, Q̄)
    prior_μ̄ = prior_tangent isa Tangent ? prior_tangent.mean : NoTangent()
    prior_Q̄_existing = prior_tangent isa Tangent ? prior_tangent.precision : NoTangent()
    combined_Q̄ = _is_zero_tangent(prior_Q̄_existing) ? Q̄ : prior_Q̄_existing + Q̄
    return Tangent{typeof(prior)}(;
        mean = prior_μ̄,
        precision = combined_Q̄,
        workspace = NoTangent(),
        constraints = NoTangent(),
        version = NoTangent()
    )
end
