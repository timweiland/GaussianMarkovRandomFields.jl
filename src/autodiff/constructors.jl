using ChainRulesCore
using SparseArrays
using LinearMaps

"""
    ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, algorithm)

Automatic differentiation rule for GMRF constructor.

This rrule enables differentiation through GMRF construction, allowing gradients to flow
back to the mean vector and precision matrix parameters. The LinearSolve algorithm is
treated as non-differentiable (NoTangent).

# Arguments
- `μ::AbstractVector`: Mean vector
- `Q::Union{AbstractMatrix, LinearMaps.LinearMap}`: Precision matrix (sparse, dense, or LinearMap)
- `algorithm`: LinearSolve algorithm (treated as non-differentiable)

# Returns
- Forward: Constructed GMRF
- Backward: Pullback function that extracts tangents for μ and Q
"""
function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, algorithm)
    # Forward computation - construct GMRF using existing implementation
    x = GMRF(μ, Q, algorithm)
    
    function GMRF_pullback(x̄)
        # Extract tangents from the GMRF tangent
        μ̄ = x̄.mean
        Q̄ = x̄.precision
        
        return NoTangent(), μ̄, Q̄, NoTangent()
    end
    
    return x, GMRF_pullback
end

"""
    ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap})

Automatic differentiation rule for GMRF constructor with default algorithm.

This handles the case where no LinearSolve algorithm is provided (uses default CholeskyFactorization).
"""
function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap})
    # Forward computation - construct GMRF using existing implementation
    x = GMRF(μ, Q)
    
    function GMRF_pullback(x̄)
        # Extract tangents from the GMRF tangent
        μ̄ = x̄.mean
        Q̄ = x̄.precision
        
        return NoTangent(), μ̄, Q̄
    end
    
    return x, GMRF_pullback
end

