using ChainRulesCore
using LinearAlgebra

"""
    _is_zero_tangent(x) -> Bool

Check if a tangent is effectively zero (nothing, NoTangent, or ZeroTangent).
"""
_is_zero_tangent(::Nothing) = true
_is_zero_tangent(::ChainRulesCore.NoTangent) = true
_is_zero_tangent(::ChainRulesCore.ZeroTangent) = true
_is_zero_tangent(::Any) = false

"""
    _add_namedtuples(nt1, nt2) -> Union{NamedTuple, Nothing, NoTangent}

Add two NamedTuples (or nothing/NoTangent) with smart handling of all cases.

**Top-level handling:**
- If both arguments are `nothing` or `NoTangent`, return `nothing` or `NoTangent`
- If one argument is `nothing` or `NoTangent`, return the other argument
- If both are NamedTuples, proceed to key-wise addition

**Key-wise addition (when both are NamedTuples):**
- If one value is `nothing` or `NoTangent`, use the other value
- If both values are non-`nothing`, add them together
- If both values are `nothing` or `NoTangent`, the result is `nothing`

# Arguments
- `nt1`: First NamedTuple, Nothing, or NoTangent
- `nt2`: Second NamedTuple, Nothing, or NoTangent (must have same keys as `nt1` if both are NamedTuples)

# Returns
- Result with smart combination of inputs

# Examples
```julia
# Top-level handling
_add_namedtuples(nothing, nothing) == nothing
_add_namedtuples(NoTangent(), (a=1,)) == (a=1,)
_add_namedtuples((a=1,), nothing) == (a=1,)

# Key-wise addition
nt1 = (a = 1.0, b = nothing, c = [1, 2])
nt2 = (a = 2.0, b = 3.0, c = [3, 4])
result = _add_namedtuples(nt1, nt2)
# result = (a = 3.0, b = 3.0, c = [4, 6])
```
"""
function _add_namedtuples(nt1::Union{NamedTuple, Nothing}, nt2::Union{NamedTuple, Nothing})
    # Handle top-level nothing cases first
    if nt1 === nothing && nt2 === nothing
        return nothing
    elseif nt1 === nothing
        return nt2
    elseif nt2 === nothing
        return nt1
    end

    # Create result by combining values for each key
    result_pairs = map(keys(nt1)) do key
        val1 = getproperty(nt1, key)
        val2 = getproperty(nt2, key)

        # Handle different combinations of nothing/non-nothing
        if val1 === nothing && val2 === nothing
            combined_val = nothing
        elseif val1 === nothing
            combined_val = val2
        elseif val2 === nothing
            combined_val = val1
        else
            # Both are non-nothing, add them
            combined_val = val1 + val2
        end

        key => combined_val
    end

    return NamedTuple(result_pairs)
end

# Handle NoTangent cases
_add_namedtuples(::ChainRulesCore.NoTangent, ::ChainRulesCore.NoTangent) = ChainRulesCore.NoTangent()
_add_namedtuples(::ChainRulesCore.NoTangent, nt::NamedTuple) = nt
_add_namedtuples(nt::NamedTuple, ::ChainRulesCore.NoTangent) = nt
_add_namedtuples(::ChainRulesCore.NoTangent, ::Nothing) = nothing
_add_namedtuples(::Nothing, ::ChainRulesCore.NoTangent) = nothing

# Handle Tangent cases
_add_namedtuples(t::ChainRulesCore.Tangent, ::ChainRulesCore.NoTangent) = t
_add_namedtuples(::ChainRulesCore.NoTangent, t::ChainRulesCore.Tangent) = t
_add_namedtuples(t1::ChainRulesCore.Tangent, t2::ChainRulesCore.Tangent) = t1 + t2

"""
    ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(gaussian_approximation),
                         prior_gmrf::Union{GMRF, ConstrainedGMRF}, obs_lik::ObservationLikelihood; kwargs...)

Backend-agnostic automatic differentiation rule for `gaussian_approximation` using the Implicit Function Theorem.

This rrule enables differentiation through the Fisher scoring optimization that finds the mode
of the posterior distribution. It works with any AD backend that supports reverse mode (Zygote,
ReverseDiff, Enzyme, etc.) via ChainRulesCore's `rrule_via_ad`.

# Mathematical Approach
Uses the Implicit Function Theorem on the optimality condition ∇ₓ neg_log_posterior(x*) = 0:
- Forward: Solve via Fisher scoring (handled by forward pass)
- Backward: Differentiate through the optimality condition:
  1. Compute λ = H⁻¹(x*) · ȳ_mean where H is the Hessian at mode
  2. VJP through ∇ₓ_neg_log_posterior gives gradients w.r.t. prior & likelihood
  3. Combine with direct ȳ_precision contribution

# Arguments
- `config::RuleConfig{>:HasReverseMode}`: AD backend configuration
- `prior_gmrf`: Prior GMRF or ConstrainedGMRF
- `obs_lik`: Observation likelihood
- `kwargs...`: Convergence parameters (max_iter, mean_change_tol, etc.) - treated as non-differentiable

# Returns
- Forward: Result of gaussian_approximation
- Pullback: Function that computes VJPs for prior_gmrf and obs_lik
"""
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(gaussian_approximation),
        prior_gmrf::Union{GMRF, ConstrainedGMRF},
        obs_lik::ObservationLikelihood;
        kwargs...
    )
    # === Forward pass ===
    posterior = gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
    x_star = mean(posterior)

    # === Pullback ===
    function gaussian_approximation_pullback(ȳ)
        # Extract tangent components for the posterior GMRF
        μ̄ = ȳ.mean
        Q̄ = ȳ.precision

        # Handle precision tangent through loghessian to get indirect x* dependence
        # The Hessian appears in the posterior precision: Q_post = Q_prior - loghessian(x*)
        # When differentiating w.r.t. θ: ∂Q_post/∂θ = ∂Q_prior/∂θ - ∂loghessian/∂x* · ∂x*/∂θ - ∂loghessian/∂obs_lik · ∂obs_lik/∂θ
        # Compute indirect x* contribution from precision gradient via loghessian
        if _is_zero_tangent(Q̄)
            # No precision tangent, only mean path contributes
            x_tangent_from_hess = nothing
            obs_lik_tangent_from_Q̄ = NoTangent()
        else
            _, hess_pullback = rrule_via_ad(config, loghessian, x_star, obs_lik)
            _, x_tangent_from_hess, obs_lik_tangent_from_Q̄ = hess_pullback(-Q̄)  # Pass -Q̄ because Q_post = Q_prior - H
        end

        # Solve for λ combining BOTH mean tangent and indirect x* dependence from precision
        # H · λ = μ̄ + x_tangent_from_hess
        # This accounts for: ∂L/∂x* from direct (mean) path + indirect (precision→hessian→x*) path
        cache = deepcopy(linsolve_cache(posterior))
        cache.b = _is_zero_tangent(x_tangent_from_hess) ? collect(μ̄) : collect(μ̄) .+ collect(x_tangent_from_hess)
        λ = solve!(cache).u

        # VJP through ∇ₓ_neg_log_posterior at x* to get gradients w.r.t. prior and likelihood
        # IFT gives: dx*/dθ = -H⁻¹ · ∂∇/∂θ, so in the pullback we need -λ
        _, ∇_pullback = rrule_via_ad(config, ∇ₓ_neg_log_posterior, prior_gmrf, obs_lik, x_star)
        _, prior_gmrf_tangent, obs_lik_tangent, _ = ∇_pullback(-λ)  # Note the minus sign from IFT

        # Add contribution from ȳ.precision to prior_gmrf tangent
        if !_is_zero_tangent(Q̄)
            # Extract mean tangent (from VJP) and combine with Q̄
            prior_μ̄ = prior_gmrf_tangent isa Tangent ? prior_gmrf_tangent.mean : NoTangent()
            prior_Q̄_existing = prior_gmrf_tangent isa Tangent ? prior_gmrf_tangent.precision : NoTangent()

            # Combine precision gradients: add direct Q̄ to existing gradient from VJP
            combined_Q̄ = _is_zero_tangent(prior_Q̄_existing) ? Q̄ : prior_Q̄_existing + Q̄

            prior_gmrf_tangent = Tangent{typeof(prior_gmrf)}(
                mean = prior_μ̄,
                precision = combined_Q̄,
                information_vector = NoTangent(),
                Q_sqrt = NoTangent(),
                linsolve_cache = NoTangent(),
                rbmc_strategy = NoTangent()
            )
        end

        # Combine tangents from mean path and precision path
        obs_lik_combined = _add_namedtuples(obs_lik_tangent, obs_lik_tangent_from_Q̄)

        # Return tangents: NoTangent for function and kwargs
        return (NoTangent(), prior_gmrf_tangent, obs_lik_combined)
    end

    return posterior, gaussian_approximation_pullback
end

# Also handle case without RuleConfig for simpler usage
function ChainRulesCore.rrule(
        ::typeof(gaussian_approximation),
        prior_gmrf::Union{GMRF, ConstrainedGMRF},
        obs_lik::ObservationLikelihood;
        kwargs...
    )
    # Delegate to the RuleConfig version with default config
    return rrule(NoRuleConfig(), gaussian_approximation, prior_gmrf, obs_lik; kwargs...)
end
