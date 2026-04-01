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

# --- gaussian_approximation rrule for WorkspaceGMRF ---

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(gaussian_approximation),
        prior_gmrf::WorkspaceGMRF,
        obs_lik::ObservationLikelihood;
        kwargs...
    )
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

        if !_is_zero_tangent(Q̄)
            prior_tangent = _workspace_add_precision_tangent(prior_tangent, prior_gmrf, Q̄)
        end

        obs_lik_combined = _add_namedtuples(obs_lik_tangent, obs_lik_tangent_from_Q̄)

        return (NoTangent(), prior_tangent, obs_lik_combined)
    end

    return posterior, workspace_ga_pullback
end

# Also handle case without RuleConfig
function ChainRulesCore.rrule(
        ::typeof(gaussian_approximation),
        prior_gmrf::WorkspaceGMRF,
        obs_lik::ObservationLikelihood;
        kwargs...
    )
    return rrule(NoRuleConfig(), gaussian_approximation, prior_gmrf, obs_lik; kwargs...)
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
