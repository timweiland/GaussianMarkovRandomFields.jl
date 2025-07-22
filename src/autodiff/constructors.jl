using ChainRulesCore
using SparseArrays
using LinearMaps

"""
    ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, solver_blueprint::AbstractSolverBlueprint)

Automatic differentiation rule for GMRF constructor.

This rrule enables differentiation through GMRF construction, allowing gradients to flow
back to the mean vector and precision matrix parameters. The solver construction is
treated as non-differentiable (NoTangent).

# Arguments
- `μ::AbstractVector`: Mean vector
- `Q::Union{AbstractMatrix, LinearMaps.LinearMap}`: Precision matrix (sparse, dense, or LinearMap)
- `solver_blueprint::AbstractSolverBlueprint`: Solver blueprint (treated as non-differentiable)

# Returns
- Forward: Constructed GMRF
- Backward: Pullback function that extracts tangents for μ and Q
"""
function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, solver_blueprint::AbstractSolverBlueprint)
    # Forward computation - construct GMRF using existing implementation
    x = GMRF(μ, Q, solver_blueprint)
    
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

Automatic differentiation rule for GMRF constructor with default solver blueprint.

This handles the case where no solver blueprint is provided (uses DefaultSolverBlueprint()).
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

