using LinearMaps, LinearAlgebra, SparseArrays

export ConstantMeshSTGMRF, ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF

################################################################################
#
#    ConstantMeshSTGMRF
#
#    A spatiotemporal GMRF with constant spatial discretization.
#
################################################################################
"""
    ConstantMeshSTGMRF

A spatiotemporal GMRF with constant spatial discretization.
"""
abstract type ConstantMeshSTGMRF{D, T, PrecisionMap<:LinearMap{T}} <: AbstractSpatiotemporalGMRF{T, PrecisionMap} end

precision_map(x::ConstantMeshSTGMRF) = x.precision
mean(x::ConstantMeshSTGMRF) = x.mean

"""
    ImplicitEulerConstantMeshSTGMRF

A spatiotemporal GMRF with constant spatial discretization and an implicit Euler
discretization of the temporal dynamics.
"""
struct ImplicitEulerConstantMeshSTGMRF{D, T, PrecisionMap<:LinearMap{T}, Solver<:AbstractSolver} <: ConstantMeshSTGMRF{D, T, PrecisionMap}
    mean::AbstractVector{T}
    precision::PrecisionMap
    discretization::FEMDiscretization{D}
    ssm::ImplicitEulerSSM
    solver::Solver

    function ImplicitEulerConstantMeshSTGMRF(
        mean::AbstractVector{T},
        precision::LinearMap{T},
        discretization::FEMDiscretization{D},
        ssm::ImplicitEulerSSM,
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {D,T}
        n = length(mean)
        n == Base.size(precision, 1) == Base.size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))

        solver = construct_solver(solver_blueprint, mean, precision)
        result = new{D,T,typeof(precision),typeof(solver)}(mean, precision, discretization, ssm, solver)
        postprocess!(solver, result)
        return result
    end
end

"""
    ConcreteConstantMeshSTGMRF

A concrete implementation of a spatiotemporal GMRF with constant spatial
discretization.
"""
struct ConcreteConstantMeshSTGMRF{D, T, PrecisionMap<:LinearMap{T}, Solver<:AbstractSolver} <: ConstantMeshSTGMRF{D, T, PrecisionMap}
    mean::AbstractVector{T}
    precision::PrecisionMap
    discretization::FEMDiscretization{D}
    solver::Solver

    function ConcreteConstantMeshSTGMRF(
        mean::AbstractVector{T},
        precision::LinearMap{T},
        discretization::FEMDiscretization{D},
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {D,T}
        n = length(mean)
        n == Base.size(precision, 1) == Base.size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))

        solver = construct_solver(solver_blueprint, mean, precision)
        result = new{D,T,typeof(precision),typeof(solver)}(mean, precision, discretization, solver)
        postprocess!(solver, result)
        return result
    end
end

N_spatial(x::ConstantMeshSTGMRF) = ndofs(x.discretization)
N_t(x::ConstantMeshSTGMRF) = length(x) ÷ N_spatial(x)

time_means(x::ConstantMeshSTGMRF) = make_chunks(mean(x), N_t(x))
time_vars(x::ConstantMeshSTGMRF) = make_chunks(var(x), N_t(x))
time_stds(x::ConstantMeshSTGMRF) = make_chunks(std(x), N_t(x))
time_rands(x::ConstantMeshSTGMRF, rng::AbstractRNG) = make_chunks(rand(rng, x), N_t(x))
discretization_at_time(x::ConstantMeshSTGMRF, ::Int) = x.discretization

################################################################################
#
# Linear conditional ConstantMeshSTGMRF
#
################################################################################
N_spatial(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = ndofs(x.prior.discretization)
N_t(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = length(x) ÷ N_spatial(x)

time_means(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(mean(x), N_t(x))
time_vars(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(var(x), N_t(x))
time_stds(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(std(x), N_t(x))
time_rands(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}, rng::AbstractRNG) =
    make_chunks(rand(rng, x), N_t(x))
discretization_at_time(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}, ::Int) =
    x.prior.discretization


function default_preconditioner_strategy(
    x::Union{<:ConstantMeshSTGMRF,LinearConditionalGMRF{<:ConstantMeshSTGMRF}},
)
    block_size = N_spatial(x)
    Q = sparse(to_matrix(precision_map(x)))
    return temporal_block_gauss_seidel(Q, block_size)
end
