using ChainRulesCore
using LinearAlgebra
using CliqueTrees.Multifrontal.Differential: ldivsym

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

# Extract μ̄ and Q̄ from the posterior tangent, handling both GMRF and ConstrainedGMRF.
# These are the tangents for the BASE GMRF's mean (= x*, the mode) and precision.
_extract_posterior_tangents(ȳ, ::GMRF) = (ȳ.mean, ȳ.precision)
function _extract_posterior_tangents(ȳ, posterior::ConstrainedGMRF)
    μ̄ = NoTangent()
    Q̄ = NoTangent()

    # The constrained_mean tangent must be projected back via P^T (not P).
    # P = I - Ã'S⁻¹A, so P^T = I - A^TS⁻¹Ã (these differ when Q ≠ I).
    if !_is_zero_tangent(ȳ.constrained_mean)
        μ̄_c = collect(ȳ.constrained_mean)
        v = posterior.L_c \ (posterior.A_tilde_T' * μ̄_c)  # S⁻¹ * Ã * μ̄_c
        μ̄ = μ̄_c - posterior.constraint_matrix' * v  # P^T * μ̄_c
    end

    # Direct base_gmrf tangent (from logpdf rrule)
    base_ȳ = ȳ.base_gmrf
    if !_is_zero_tangent(base_ȳ) && base_ȳ isa Tangent
        if !_is_zero_tangent(base_ȳ.mean)
            μ̄ = _is_zero_tangent(μ̄) ? base_ȳ.mean : μ̄ + collect(base_ȳ.mean)
        end
        if !_is_zero_tangent(base_ȳ.precision)
            Q̄ = base_ȳ.precision
        end
    end

    # log_constraint_correction tangent → flows back through correction's μ and Q dependence
    if !_is_zero_tangent(ȳ.log_constraint_correction)
        c̄ = ȳ.log_constraint_correction
        resid = posterior.constraint_vector - posterior.constraint_matrix * mean(posterior.base_gmrf)

        # ∂c/∂μ contribution
        μ̄_corr = c̄ * (posterior.constraint_matrix' * (posterior.L_c \ (-resid)))
        μ̄ = _is_zero_tangent(μ̄) ? μ̄_corr : μ̄ + μ̄_corr

        # ∂c/∂Q contribution
        S_inv = inv(posterior.L_c)
        w = S_inv * resid
        Q̄_corr = c̄ * (-0.5) * posterior.A_tilde_T * (S_inv - w * w') * posterior.A_tilde_T'
        Q̄ = _is_zero_tangent(Q̄) ? Q̄_corr : Q̄ + Q̄_corr
    end

    return (μ̄, Q̄)
end

# Combine the Q̄ contribution with the existing prior tangent from the VJP.
function _add_precision_tangent(prior_tangent, prior::GMRF, Q̄)
    prior_μ̄ = prior_tangent isa Tangent ? prior_tangent.mean : NoTangent()
    prior_Q̄_existing = prior_tangent isa Tangent ? prior_tangent.precision : NoTangent()
    combined_Q̄ = _is_zero_tangent(prior_Q̄_existing) ? Q̄ : prior_Q̄_existing + Q̄
    return Tangent{typeof(prior)}(;
        mean = prior_μ̄,
        precision = combined_Q̄,
        information_vector = NoTangent(),
        Q_sqrt = NoTangent(),
        linsolve_cache = NoTangent(),
        rbmc_strategy = NoTangent(),
    )
end

function _add_precision_tangent(prior_tangent, prior::ConstrainedGMRF, Q̄)
    # Extract base_gmrf tangent from the nested structure
    base_tangent = (prior_tangent isa Tangent && hasproperty(prior_tangent, :base_gmrf)) ?
        prior_tangent.base_gmrf : NoTangent()
    base_μ̄ = (!_is_zero_tangent(base_tangent) && base_tangent isa Tangent) ? base_tangent.mean : NoTangent()
    base_Q̄_existing = (!_is_zero_tangent(base_tangent) && base_tangent isa Tangent) ? base_tangent.precision : NoTangent()
    combined_Q̄ = _is_zero_tangent(base_Q̄_existing) ? Q̄ : base_Q̄_existing + Q̄

    new_base_tangent = Tangent{typeof(prior.base_gmrf)}(;
        mean = base_μ̄,
        precision = combined_Q̄,
        information_vector = NoTangent(),
        Q_sqrt = NoTangent(),
        linsolve_cache = NoTangent(),
        rbmc_strategy = NoTangent(),
    )

    # Preserve any existing ConstrainedGMRF-level tangents from the VJP
    return Tangent{typeof(prior)}(;
        base_gmrf = new_base_tangent,
        constraint_matrix = NoTangent(),
        constraint_vector = NoTangent(),
        A_tilde_T = NoTangent(),
        L_c = NoTangent(),
        constrained_mean = NoTangent(),
        log_constraint_correction = NoTangent(),
    )
end

# Unwrap a ConstrainedGMRF tangent through the _apply_constraints step.
# For GMRF posteriors, this is a no-op. For ConstrainedGMRF posteriors,
# it propagates the tangent back through ConstrainedGMRF(inner_gmrf, A, e)
# to get the tangent for inner_gmrf.
function _unwrap_constraints_tangent(ȳ, ::GMRF, ::Nothing, config)
    return ȳ, NoTangent()
end

function _unwrap_constraints_tangent(ȳ, posterior::ConstrainedGMRF, constraints, config)
    inner_gmrf = posterior.base_gmrf
    A = constraints.A
    e = constraints.e

    # Use rrule_via_ad to differentiate through ConstrainedGMRF(inner_gmrf, A, e)
    _, constructor_pullback = rrule_via_ad(
        config, ConstrainedGMRF, inner_gmrf, A, e
    )
    _, inner_tangent, _, _ = constructor_pullback(ȳ)

    return inner_tangent, NoTangent()
end

# Wrap a base GMRF tangent in a ConstrainedGMRF tangent if the prior is constrained.
_wrap_prior_tangent(base_tangent, ::GMRF) = base_tangent
_wrap_prior_tangent(base_tangent, ::ChordalGMRF) = base_tangent
function _wrap_prior_tangent(base_tangent, prior::ConstrainedGMRF)
    return Tangent{typeof(prior)}(;
        base_gmrf = base_tangent,
        constraint_matrix = NoTangent(),
        constraint_vector = NoTangent(),
        A_tilde_T = NoTangent(),
        L_c = NoTangent(),
        constrained_mean = NoTangent(),
        log_constraint_correction = NoTangent(),
    )
end

# Combine x* tangent contributions from the mean and precision paths, handling zero tangents.
function _combine_x_tangents(x̄_from_μ, x̄_from_Q, x_star)
    if _is_zero_tangent(x̄_from_μ) && _is_zero_tangent(x̄_from_Q)
        return zeros(eltype(x_star), length(x_star))
    elseif _is_zero_tangent(x̄_from_μ)
        return collect(x̄_from_Q)
    elseif _is_zero_tangent(x̄_from_Q)
        return collect(x̄_from_μ)
    else
        return collect(x̄_from_μ) .+ collect(x̄_from_Q)
    end
end

# IFT linear solve: solve Q_post * λ = x̄_total, with constraint projection for GMRF/ConstrainedGMRF.
function _ift_solve(posterior::Union{GMRF, ConstrainedGMRF}, x̄_total, prior_gmrf)
    cache = linsolve_cache(_base_gmrf(posterior))
    b_saved = copy(cache.b)
    cache.b = x̄_total
    λ = copy(solve!(cache).u)
    λ = _constrain_step(λ, cache, _extract_constraints(prior_gmrf))
    cache.b .= b_saved
    return λ
end

function _ift_solve(posterior::ChordalGMRF, x̄_total, ::ChordalGMRF)
    return ldivsym(precision_matrix(posterior), posterior.L, posterior.P, x̄_total)
end

"""
    ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(gaussian_approximation),
                         prior_gmrf::Union{GMRF, ConstrainedGMRF, ChordalGMRF},
                         obs_lik::ObservationLikelihood; kwargs...)

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
- `prior_gmrf`: Prior GMRF, ConstrainedGMRF, or ChordalGMRF
- `obs_lik`: Observation likelihood
- `kwargs...`: Convergence parameters (max_iter, mean_change_tol, etc.) - treated as non-differentiable

# Returns
- Forward: Result of gaussian_approximation
- Pullback: Function that computes VJPs for prior_gmrf and obs_lik
"""
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(gaussian_approximation),
        prior_gmrf::Union{GMRF, ConstrainedGMRF, ChordalGMRF},
        obs_lik::ObservationLikelihood;
        kwargs...
    )
    # === Forward pass ===
    posterior = gaussian_approximation(prior_gmrf, obs_lik; kwargs...)
    x_star = mean(posterior)

    # === Pullback ===
    function gaussian_approximation_pullback(ȳ)
        if ȳ isa ZeroTangent
            return (NoTangent(), ZeroTangent(), NoTangent())
        end

        # Extract tangent components — dispatches on posterior type
        μ̄, Q̄ = _extract_posterior_tangents(ȳ, posterior)

        # Q̄ path: backprop through Q_post = Q_prior - loghessian(x*, obs_lik)
        if _is_zero_tangent(Q̄)
            x̄_from_Q = ZeroTangent()
            obs_lik_tangent_from_Q = NoTangent()
        else
            _, hess_pullback = rrule_via_ad(config, loghessian, x_star, obs_lik)
            _, x̄_from_Q, obs_lik_tangent_from_Q = hess_pullback(-Q̄)
        end

        # μ̄ path: μ_post = x*, so μ̄ flows directly to x*
        x̄_from_μ = _is_zero_tangent(μ̄) ? ZeroTangent() : μ̄

        # Combine x* tangents and solve Q_post * λ = x̄_total via IFT
        x̄_total = _combine_x_tangents(x̄_from_μ, x̄_from_Q, x_star)
        λ = _ift_solve(posterior, x̄_total, prior_gmrf)

        # VJP through ∇ₓ_neg_log_posterior at x* to get gradients w.r.t. prior and likelihood
        base_prior = _base_gmrf(prior_gmrf)
        _, ∇_pullback = rrule_via_ad(config, ∇ₓ_neg_log_posterior, base_prior, obs_lik, x_star)
        _, base_prior_tangent, obs_lik_tangent, _ = ∇_pullback(-λ)

        # Add contribution from ȳ.precision to base prior tangent
        if !_is_zero_tangent(Q̄)
            base_prior_tangent = _add_precision_tangent(base_prior_tangent, base_prior, Q̄)
        end

        # Wrap the base GMRF tangent in a ConstrainedGMRF tangent if needed
        prior_gmrf_tangent = _wrap_prior_tangent(base_prior_tangent, prior_gmrf)

        # Combine tangents from mean path and precision path
        obs_lik_combined = _add_namedtuples(obs_lik_tangent, obs_lik_tangent_from_Q)

        return (NoTangent(), prior_gmrf_tangent, obs_lik_combined)
    end

    return posterior, gaussian_approximation_pullback
end

# =============================================================================
# ChordalGMRF tangent helpers (dispatched from unified rrule above)
# =============================================================================

# Extract tangents from ChordalGMRF posterior
function _extract_posterior_tangents(ȳ, ::ChordalGMRF)
    μ̄ = _is_zero_tangent(ȳ.μ) ? ZeroTangent() : ȳ.μ
    Q̄ = _is_zero_tangent(ȳ.Q) ? ZeroTangent() : ȳ.Q
    return (μ̄, Q̄)
end

# Add Q̄ contribution to prior tangent for ChordalGMRF
function _add_precision_tangent(prior_tangent, prior::ChordalGMRF, Q̄)
    prior_μ̄ = (prior_tangent isa Tangent && hasproperty(prior_tangent, :μ)) ? prior_tangent.μ : NoTangent()
    prior_Q̄_existing = (prior_tangent isa Tangent && hasproperty(prior_tangent, :Q)) ? prior_tangent.Q : NoTangent()
    combined_Q̄ = _is_zero_tangent(prior_Q̄_existing) ? Q̄ : prior_Q̄_existing + Q̄
    return Tangent{typeof(prior)}(;
        μ = prior_μ̄,
        Q = combined_Q̄,
        L = NoTangent(),
        P = NoTangent(),
    )
end
