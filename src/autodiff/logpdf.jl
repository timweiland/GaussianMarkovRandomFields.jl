using ChainRulesCore
using Distributions: logpdf
using LinearAlgebra
using SparseArrays
using SelectedInversion

"""
    ChainRulesCore.rrule(::typeof(logpdf), x::AbstractGMRF, z::AbstractVector)

Efficient reverse-mode automatic differentiation rule for GMRF logpdf computation.

This implementation uses selected inverses via SelectedInversion.jl to compute gradients
efficiently without materializing the full covariance matrix. Works with any Cholesky
factorization backend (CHOLMOD, Pardiso).

# Forward computation
Uses the existing logpdf implementation from Distributions.jl

# Backward computation
- ∂logpdf/∂μ = Q(z - μ)
- ∂logpdf/∂Q = -0.5 * (Q⁻¹ + (z - μ)(z - μ)ᵀ)

The key insight is using `selinv(chol)` to compute only the nonzero entries of Q⁻¹
efficiently, avoiding the O(n³) cost of full matrix inversion.
"""
function ChainRulesCore.rrule(::typeof(logpdf), x::AbstractGMRF, z::AbstractVector)
    μ = mean(x)
    Q = precision_matrix(x)
    r = z - μ

    # Check if GMRF supports selected inversion for efficient gradients
    if hasproperty(x, :linsolve_cache) && supports_selinv(x.linsolve_cache.alg) == Val{true}()
        # Forward computation - use existing implementation
        val = logpdf(x, z)

        function logpdf_pullback(ȳ)
            # Compute selected inverse efficiently using our dispatch system
            Qinv = selinv(x.linsolve_cache)
            Qr = Q * r

            # Gradients
            μ̄ = ȳ * Qr                    # ∂logpdf/∂μ = Q(z - μ) → chain rule

            # Compute gradient w.r.t. precision matrix
            # Uses dispatch to handle different matrix types from selinv
            Q̄ = compute_precision_gradient(Qinv, r, ȳ)

            # Tangent for GMRF - use new structure
            x̄ = Tangent{typeof(x)}(
                linsolve_cache = NoTangent(),  # LinearSolve cache is not differentiable
                mean = μ̄,                     # Chain rule propagates μ̄ = ȳ * Q(z - μ)
                precision = Q̄,                # ∂logpdf/∂Q
                information_vector = NoTangent(),
                Q_sqrt = NoTangent(),
                rbmc_strategy = NoTangent()
            )

            # Tangent for observation vector z
            z̄ = ȳ * (-Qr)                    # ∂logpdf/∂z = -Q(z - μ)

            return NoTangent(), x̄, z̄
        end

        return val, logpdf_pullback
    else
        error(
            "Automatic differentiation through logpdf requires an algorithm that supports selected inversion " *
                "(CHOLMODFactorization, CholeskyFactorization). " *
                "Got algorithm type: $(typeof(x.linsolve_cache.alg)). " *
                "Consider using LinearSolve.CHOLMODFactorization() or LinearSolve.CholeskyFactorization()."
        )
    end
end
