# COV_EXCL_START
"""
    GaussianMarkovRandomFieldsEnzyme

Package extension providing Enzyme.jl support for automatic differentiation in GaussianMarkovRandomFields.jl.

This extension implements custom EnzymeRules for efficient reverse-mode automatic differentiation
through GMRF operations. It provides specialized rules for:

- **GMRF construction**: Differentiate through `GMRF(μ, Q)` and `GMRF(μ, Q, algorithm)`
- **logpdf computation**: Efficient gradients using selected inversion for sparse precision matrices
- **gaussian_approximation**: Implicit Function Theorem-based gradients through Fisher scoring optimization

These custom rules enable gradient-based inference and optimization with Enzyme.jl, complementing
the ChainRulesCore-based rules for Zygote and other AD backends.

# Performance Notes
The Enzyme rules use the same mathematical approach as the ChainRulesCore rules but are implemented
directly in Enzyme's custom rule system for optimal performance with Enzyme's compilation pipeline.

# Example
```julia
using GaussianMarkovRandomFields
using Enzyme

# Define objective function
function objective(θ)
    μ, Q = build_gmrf_params(θ)
    gmrf = GMRF(μ, Q)
    return logpdf(gmrf, data)
end

# Compute gradient with Enzyme
grad = autodiff(Reverse, objective, Active, Duplicated(θ, zero(θ)))[1]
```
"""
module GaussianMarkovRandomFieldsEnzyme

using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: GMRF, AbstractGMRF, precision_matrix, selinv,
    loghessian, ∇ₓ_neg_log_posterior, linsolve_cache,
    gaussian_approximation, ObservationLikelihood,
    compute_precision_gradient
using Enzyme
using LinearMaps
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using LinearSolve

# Import Enzyme's custom rule functions
using Enzyme.EnzymeRules
import Enzyme: Const, Active, Duplicated, Annotation

"""
Helper function to accumulate gradient into precision matrix, handling Symmetric wrapper.

Uses multiple dispatch to handle regular matrices vs Symmetric-wrapped matrices.
"""
accumulate_precision_gradient!(target::AbstractMatrix, gradient) = target .+= gradient
accumulate_precision_gradient!(target::Symmetric, gradient::Symmetric) = target.data .+= gradient.data
accumulate_precision_gradient!(target::Symmetric, gradient) = target.data .+= gradient
function accumulate_precision_gradient!(target::SymTridiagonal, gradient)
    target.dv .+= diag(gradient)
    return target.ev .+= diag(gradient, -1) + diag(gradient, 1)
end
function accumulate_precision_gradient!(target::SymTridiagonal, gradient::SymTridiagonal)
    target.dv .+= gradient.dv
    return target.ev .+= gradient.ev
end


"""
Custom Enzyme rule for GMRF constructor.

The GMRF constructor is a simple wrapper that stores μ and Q without modification,
so the gradient flow is straightforward:
- ∂L/∂μ comes from the .mean field of the GMRF tangent
- ∂L/∂Q comes from the .precision field of the GMRF tangent
"""
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Enzyme.Const{Type{GMRF}},
        ::Type{RT},
        μ::Annotation{MT},
        Q::Annotation{QT}
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    primal = func.val(μ.val, Q.val)

    if Enzyme.EnzymeRules.needs_shadow(config)
        mu_shadow = μ isa Duplicated ? copy(μ.dval) : zero(μ.val)
        Q_shadow = Q isa Duplicated ? copy(Q.dval) : zero(Q.val)
        dres = func.val(mu_shadow, Q_shadow)
    else
        dres = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, dres, dres)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Enzyme.Const{Type{GMRF}},
        ::Type{RT},
        tape,
        μ::Annotation{MT},
        Q::Annotation{QT}
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    mu_shadow, Q_shadow = mean(tape), precision_matrix(tape)

    # Accumulate gradients into input shadows
    if μ isa Duplicated
        μ.dval .+= mu_shadow
        fill!(mu_shadow, 0.0)
    end
    if Q isa Duplicated
        Q.dval .+= Q_shadow
        fill!(Q_shadow, 0.0)
    end

    return (nothing, nothing)
end

"""
Custom Enzyme rule for GMRF constructor with algorithm argument.

Handles GMRF(μ, Q, algorithm) where algorithm is non-differentiable.
"""
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        μ::Annotation{MT},
        Q::Annotation{QT},
        alg::Const
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    primal = func.val(μ.val, Q.val, alg.val)

    if Enzyme.EnzymeRules.needs_shadow(config)
        mu_shadow = μ isa Duplicated ? copy(μ.dval) : zero(μ.val)
        Q_shadow = Q isa Duplicated ? copy(Q.dval) : zero(Q.val)
        dres = func.val(mu_shadow, Q_shadow, alg.val)
    else
        dres = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, dres, dres)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        tape,
        μ::Annotation{MT},
        Q::Annotation{QT},
        alg::Const
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    mu_shadow, Q_shadow = mean(tape), precision_matrix(tape)

    # Accumulate gradients into input shadows
    if μ isa Duplicated
        μ.dval .+= mu_shadow
        fill!(mu_shadow, 0.0)
    end
    if Q isa Duplicated
        Q.dval .+= Q_shadow
        fill!(Q_shadow, 0.0)
    end

    return (nothing, nothing, nothing)
end

"""
Custom Enzyme rule for logpdf evaluation of GMRF.

Computes gradients using selected inversion for efficient sparse precision matrix handling:
- ∂logpdf/∂μ = Q(z - μ)
- ∂logpdf/∂Q = 0.5 * (Q⁻¹ - (z-μ)(z-μ)ᵀ)
- ∂logpdf/∂z = -Q(z - μ)
"""
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logpdf)},
        ::Type{RT},
        x::Annotation{<:AbstractGMRF},
        z::Annotation{<:AbstractVector}
    ) where {RT}
    # Forward computation
    primal = logpdf(x.val, z.val)

    # Save values needed for reverse pass
    μ = mean(x.val)
    Q = precision_matrix(x.val)
    r = z.val - μ

    # Compute Q⁻¹ efficiently using selected inversion
    Qinv = selinv(x.val.linsolve_cache)

    # Save to tape
    tape = (Q, r, Qinv)

    if Enzyme.EnzymeRules.needs_shadow(config)
        shadow = zero(typeof(primal))
    else
        shadow = nothing
    end

    # Return primal only if needed
    primal_out = EnzymeRules.needs_primal(config) ? primal : nothing

    return EnzymeRules.AugmentedReturn(primal_out, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(logpdf)},
        dret::Active,
        tape,
        x::Annotation{<:AbstractGMRF},
        z::Annotation{<:AbstractVector}
    )
    # Extract saved values and incoming gradient
    ȳ = dret.val
    Q, r, Qinv = tape

    # Compute Q * r (used for both mean and z gradients)
    Qr = Q * r

    # Gradient w.r.t. GMRF (mean and precision)
    if x isa Duplicated
        # ∂logpdf/∂μ = Q(z - μ)
        x.dval.mean .+= ȳ * Qr

        # ∂logpdf/∂Q = 0.5 * (Q⁻¹ - (z-μ)(z-μ)ᵀ)
        # Uses dispatch to handle different matrix types from selinv
        Q̄ = compute_precision_gradient(Qinv, r, ȳ)
        accumulate_precision_gradient!(x.dval.precision, Q̄)
    end

    # Gradient w.r.t. observation vector z
    if z isa Duplicated
        # ∂logpdf/∂z = -Q(z - μ)
        z.dval .+= ȳ * (-Qr)
    end

    return (nothing, nothing)
end

"""
Custom Enzyme rule for gaussian_approximation using Implicit Function Theorem.

The gaussian_approximation finds x* such that ∇ₓ neg_log_posterior(x*) = 0.
Using IFT: dx*/dθ = -H⁻¹ · ∂∇/∂θ where H is the Hessian.
"""
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        prior_gmrf::Annotation,
        obs_lik::Annotation
    ) where {RT}
    # Forward pass
    posterior = gaussian_approximation(prior_gmrf.val, obs_lik.val)
    x_star = mean(posterior)

    # Create shadow if needed
    if Enzyme.EnzymeRules.needs_shadow(config)
        shadow = deepcopy(posterior)
        shadow.mean .= zero(eltype(shadow.mean))
        shadow.precision .= zero(eltype(shadow.precision))
    else
        shadow = nothing
    end

    # Save everything to tape, including shadow
    tape = (x_star = x_star, posterior = posterior, shadow = shadow)

    return EnzymeRules.AugmentedReturn(posterior, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        tape,
        prior_gmrf::Annotation,
        obs_lik::Annotation
    ) where {RT}
    x_star = tape.x_star
    posterior = tape.posterior
    shadow = tape.shadow

    # Extract shadows from tape (not from dret - it's a Type!)
    μ̄ = shadow.mean
    Q̄ = shadow.precision

    # Check if precision gradient is non-zero
    has_Q_grad = !all(iszero, Q̄)

    # Compute indirect x* contribution from precision gradient via loghessian
    if has_Q_grad
        # VJP: ∂loghessian/∂x* · (-Q̄) and ∂loghessian/∂obs_lik · (-Q̄)
        x_tangent_from_hess = zero(x_star)

        function loghessian_vjp(x, ol)
            H = loghessian(x, ol)
            return sum((-Q̄) .* H)  # Dot product with seed
        end


        Enzyme.autodiff(
            Reverse, Const(loghessian_vjp),
            Active,
            Duplicated(copy(x_star), x_tangent_from_hess),
            obs_lik
        )
    else
        x_tangent_from_hess = zero(x_star)
    end

    # Solve H·λ = μ̄ + x_tangent_from_hess
    cache = deepcopy(linsolve_cache(posterior))
    cache.b = collect(μ̄) .+ x_tangent_from_hess
    λ = solve!(cache).u

    # VJP through ∇ₓ_neg_log_posterior with seed -λ
    prior_gmrf_shadow = deepcopy(prior_gmrf.val)
    # Zero out the shadow
    prior_gmrf_shadow.mean .= 0
    prior_gmrf_shadow.precision .= 0

    function neg_log_posterior_vjp(pg, ol, x)
        g = ∇ₓ_neg_log_posterior(pg, ol, x)
        return dot(-λ, g)
    end

    Enzyme.autodiff(
        Reverse, Const(neg_log_posterior_vjp),
        Active,
        Duplicated(prior_gmrf.val, prior_gmrf_shadow),
        obs_lik,
        Const(x_star)
    )

    # Accumulate gradients
    if prior_gmrf isa Duplicated
        # Add contributions from VJP
        prior_gmrf.dval.mean .+= prior_gmrf_shadow.mean
        prior_gmrf.dval.precision .+= prior_gmrf_shadow.precision

        # Add direct Q̄ contribution to precision
        if has_Q_grad
            prior_gmrf.dval.precision .+= Q̄
        end
    end

    # Note: obs_lik gradients are automatically accumulated by Enzyme in the autodiff calls above

    # Zero out return shadow
    fill!(shadow.mean, 0)
    fill!(shadow.precision, 0)

    return (nothing, nothing)
end

end
# COV_EXCL_STOP
