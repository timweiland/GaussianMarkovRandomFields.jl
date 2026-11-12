using LinearAlgebra, LinearMaps

export ADJacobianMap, ADJacobianAdjointMap

"""
    ADJacobianMap(f, x₀::AbstractVector{T}, N_outputs::Int)

A linear map representing the Jacobian of `f` at `x₀`.
Uses forward-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute JVPs.

Requires ForwardDiff.jl!

# Arguments
- `f`: Function (or other callable) to differentiate.
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
- `N_outputs::Int`: Output dimension of `f`.
"""
struct ADJacobianMap{T, F, X <: AbstractVector{T}} <: LinearMaps.LinearMap{T}
    f::F
    x₀::X
    N_outputs::Int

    function ADJacobianMap(f::F, x₀::X, N_outputs::Int) where {T, F, X <: AbstractVector{T}}
        N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))
        return new{T, F, X}(f, x₀, N_outputs)
    end
end

function LinearMaps.size(J::ADJacobianMap)
    return (J.N_outputs, length(J.x₀))
end

LinearAlgebra.adjoint(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianMap) = ADJacobianAdjointMap(J.f, J.x₀, J.N_outputs)

"""
    ADJacobianAdjointMap(f, x₀::AbstractVector{T}, N_outputs::Int)

A linear map representing the adjoint of the Jacobian of `f` at `x₀`.
Uses reverse-mode AD in a matrix-free way, i.e. we do not actually store
the Jacobian in memory and only compute VJPs.

Requires Zygote.jl!

# Arguments
- `f`: Function (or other callable) to differentiate.
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
- `N_outputs::Int`: Output dimension of `f`.
"""
struct ADJacobianAdjointMap{T, F, X <: AbstractVector{T}, V, P} <: LinearMaps.LinearMap{T}
    f::F
    x₀::X
    N_outputs::Int
    f_val::V
    f_pullback::P

    function ADJacobianAdjointMap(
            f::F, x₀::X, N_outputs::Int, f_val::V, f_pullback::P
        ) where {T, F, X <: AbstractVector{T}, V, P}
        return new{T, F, X, V, P}(f, x₀, N_outputs, f_val, f_pullback)
    end
end

# Fallback for the 3-arg `(f, x₀, N_outputs)` constructor, which is implemented in the
# Zygote extension. Vararg{Any} keeps the fallback strictly less specific than the
# extension's concrete signature (no method overwriting). The error tells the user
# what to load; JET sees a matching method so no false "no matching method" report.
function ADJacobianAdjointMap(args::Vararg{Any})
    return throw(
        ArgumentError(
            "Building an ADJacobianAdjointMap from (f, x₀, N_outputs) requires the " *
                "Zygote extension. Load Zygote to enable reverse-mode AD adjoints of ADJacobianMap."
        )
    )
end

function LinearMaps.size(J::ADJacobianAdjointMap)
    return (length(J.x₀), J.N_outputs)
end

LinearAlgebra.adjoint(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
LinearAlgebra.transpose(J::ADJacobianAdjointMap) = ADJacobianMap(J.f, J.x₀, J.N_outputs)
