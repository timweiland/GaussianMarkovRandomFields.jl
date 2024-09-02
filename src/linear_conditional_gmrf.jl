using LinearMaps, SparseArrays

export LinearConditionalGMRF

struct LinearConditionalGMRF{G<:AbstractGMRF} <: AbstractGMRF
    prior::G
    precision::LinearMap
    A::AbstractMatrix
    Q_ϵ::AbstractMatrix
    y::AbstractVector
    b::AbstractVector
    solver_ref::Base.RefValue{AbstractSolver}

    function LinearConditionalGMRF(
        prior::G,
        A::AbstractMatrix,
        Q_ϵ::AbstractMatrix,
        y::AbstractVector,
        b::AbstractVector = spzeros(size(A, 1)),
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {G}
        precision = prior.precision + LinearMap(Hermitian(A' * Q_ϵ * A))
        size(A, 1) == length(y) == length(b) || throw(ArgumentError("size mismatch"))
        solver_ref = Base.RefValue{AbstractSolver}()
        x = new{G}(prior, precision, A, Q_ϵ, y, b, solver_ref)
        solver_ref[] = construct_solver(solver_blueprint, x)
        return x
    end
end

length(d::LinearConditionalGMRF) = size(d.precision, 1)
precision_map(d::LinearConditionalGMRF) = d.precision
