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
factorization backend (CHOLMOD, LDLFactorizations, Pardiso).

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
    
    # Try to access Cholesky factorization - only works with Cholesky-based solvers
    chol = try
        x.solver.precision_chol
    catch
        error("Automatic differentiation through logpdf currently only supports Cholesky-based solvers " *
              "(CholeskySolver, LinearConditionalCholeskySolver). " *
              "Got solver type: $(typeof(x.solver)). " *
              "Consider using CholeskySolverBlueprint() or CholeskySolverBlueprint{:autodiffable}().")
    end

    # Forward computation - use existing implementation
    val = logpdf(x, z)
    
    function logpdf_pullback(ȳ)
        # Compute selected inverse efficiently
        Qinv = selinv(chol; depermute=true).Z
        Qr = Q * r
        
        # Gradients
        μ̄ = ȳ * Qr                    # ∂logpdf/∂μ = Q(z - μ) → chain rule
        
        # Compute sparse gradient w.r.t. precision matrix
        # Only compute for nonzero structure of Q⁻¹
        rows, cols, vals = findnz(Qinv)
        rr_vals = r[rows] .* r[cols]
        Q̄ = sparse(rows, cols, (0.5 * ȳ) .* (vals .- rr_vals), size(Q)...)
        
        # Tangent for GMRF - only include differentiable fields
        x̄ = Tangent{typeof(x)}(
            solver = NoTangent(),          # Solver is not differentiable
            mean = μ̄,                     # ∂L/∂μ = Q(z - μ) → ∂L/∂mean = -Q(z - μ)
            precision = Q̄                  # ∂L/∂Q
        )
        
        # Tangent for observation vector z
        z̄ = ȳ * (-Qr)                    # ∂logpdf/∂z = -Q(z - μ)
        
        return NoTangent(), x̄, z̄
    end
    
    return val, logpdf_pullback
end

