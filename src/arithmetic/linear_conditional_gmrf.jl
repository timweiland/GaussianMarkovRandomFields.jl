using LinearMaps, SparseArrays
import Base: show

export LinearConditionalGMRF

ensure_linearmap(A::AbstractMatrix) = LinearMap(A)
ensure_linearmap(A::LinearMap) = A

"""
    LinearConditionalGMRF{G}(
        prior::G,
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap},
        y::AbstractVector,
        b::AbstractVector=spzeros(size(A, 1)),
        solver_blueprint::AbstractSolverBlueprint=DefaultSolverBlueprint(),
    ) where {G<:AbstractGMRF}

A GMRF conditioned on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ)`.

# Arguments
- `prior::G`: The prior GMRF.
- `A::Union{AbstractMatrix,LinearMap}`: The matrix `A`.
- `Q_ϵ::Union{AbstractMatrix,LinearMap}`: The precision matrix of the
                                          noise term `ϵ`.
- `y::AbstractVector=spzeros(size(A, 1))`: The observations `y`.
- `b::AbstractVector=spzeros(size(A, 1))`: The offset vector `b`.
- `solver_blueprint::AbstractSolverBlueprint=DefaultSolverBlueprint()`:
        The solver blueprint.
"""
struct LinearConditionalGMRF{G<:AbstractGMRF} <: AbstractGMRF
    prior::G
    precision::LinearMap
    A::LinearMap
    Q_ϵ::LinearMap
    y::AbstractVector
    b::AbstractVector
    solver_ref::Base.RefValue{AbstractSolver}

    function LinearConditionalGMRF(
        prior::G,
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap},
        y::AbstractVector,
        b::AbstractVector = spzeros(Base.size(A, 1)),
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {G}
        A = ensure_linearmap(A)
        Q_ϵ = ensure_linearmap(Q_ϵ)
        precision = precision_map(prior) + OuterProductMap(A, Q_ϵ)
        Base.size(A, 1) == length(y) == length(b) || throw(ArgumentError("size mismatch"))
        solver_ref = Base.RefValue{AbstractSolver}()
        x = new{G}(prior, precision, A, Q_ϵ, y, b, solver_ref)
        solver_ref[] = construct_solver(solver_blueprint, x)
        return x
    end
end

length(d::LinearConditionalGMRF) = Base.size(d.precision, 1)
precision_map(d::LinearConditionalGMRF) = d.precision

function Base.show(io::IO, x::LinearConditionalGMRF)
    print(
        io,
        "LinearConditionalGMRF of size $(length(x)) and solver $(typeof(x.solver_ref[]))",
    )
end
