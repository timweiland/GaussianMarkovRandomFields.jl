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
struct LinearConditionalGMRF{
    PriorGMRF<:AbstractGMRF, 
    T<:Real, 
    PrecisionMap<:LinearMaps.LinearMap{T}, 
    ObservationMap<:LinearMaps.LinearMap, 
    NoiseMap<:LinearMaps.LinearMap, 
    Solver<:AbstractSolver
    } <: AbstractGMRF{T, PrecisionMap}
    prior::PriorGMRF
    precision::PrecisionMap
    A::ObservationMap
    Q_ϵ::NoiseMap
    y::Vector{T}
    b::Vector{T}
    solver::Solver
    #solver_ref::Base.RefValue{AbstractSolver}

    function LinearConditionalGMRF(
        prior::AbstractGMRF,
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap},
        y::AbstractVector,
        b::AbstractVector = zeros(size(A, 1)),
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    )
        A = ensure_linearmap(A)
        Q_ϵ = ensure_linearmap(Q_ϵ)
        precision = precision_map(prior) + OuterProductMap(A, Q_ϵ)
        return LinearConditionalGMRF(prior, precision, A, Q_ϵ, y, b, solver_blueprint)
    end

    function LinearConditionalGMRF(
        prior::AbstractGMRF{T},
        Q_cond::Union{AbstractMatrix, LinearMap},
        A::Union{AbstractMatrix,LinearMap},
        Q_ϵ::Union{AbstractMatrix,LinearMap},
        y::AbstractVector,
        b::AbstractVector = zeros(size(A, 1)),
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {T<:Real}
        Q_cond = ensure_linearmap(Q_cond)
        Base.size(Q_cond) == Base.size(precision_map(prior)) || throw(ArgumentError("size mismatch"))
        A = ensure_linearmap(A)
        Q_ϵ = ensure_linearmap(Q_ϵ)
        Base.size(A, 1) == length(y) == length(b) || throw(ArgumentError("size mismatch"))
        solver = construct_conditional_solver(
            solver_blueprint,
            mean(prior),
            Q_cond,
            A,
            Q_ϵ,
            y,
            b
        )
        result = new{typeof(prior), T, typeof(Q_cond), typeof(A), typeof(Q_ϵ), typeof(solver)}(prior, Q_cond, A, Q_ϵ, y, b, solver)
        postprocess!(solver, result)
        return result
    end
end

length(d::LinearConditionalGMRF) = Base.size(d.precision, 1)
precision_map(d::LinearConditionalGMRF) = d.precision

function Base.show(io::IO, x::LinearConditionalGMRF)
    print(
        io,
        "LinearConditionalGMRF of size $(length(x)) and solver $(typeof(x.solver))",
    )
end
