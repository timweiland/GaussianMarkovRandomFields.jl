using LinearMaps, LinearAlgebra, SparseArrays

export ConstantMeshSTGMRF

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
struct ConstantMeshSTGMRF{D,T} <: AbstractSpatiotemporalGMRF
    mean::AbstractVector{T}
    precision::LinearMap{T}
    discretization::FEMDiscretization{D}
    ssm::ImplicitEulerSSM
    solver_ref::Base.RefValue{AbstractSolver}

    function ConstantMeshSTGMRF(
        mean::AbstractVector{T},
        precision::LinearMap{T},
        discretization::FEMDiscretization{D},
        ssm::ImplicitEulerSSM,
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {D,T}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))

        solver_ref = Base.RefValue{AbstractSolver}()
        self = new{D,T}(mean, precision, discretization, ssm, solver_ref)
        solver_ref[] = construct_solver(solver_blueprint, self)
        return self
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

################################################################################
#
# Constrained ConstantMeshSTGMRF
#
################################################################################
function ConstrainedGMRF(
    inner_gmrf::ConstantMeshSTGMRF,
    constraint_handler::Ferrite.ConstraintHandler,
)
    x₀_constrained = ConstrainedGMRF(inner_gmrf.ssm.x₀, constraint_handler)

    prescribed_dofs_t₀ = x₀_constrained.prescribed_dofs
    free_dofs_t₀ = x₀_constrained.free_dofs
    free_to_prescribed_mat_t₀ = to_matrix(x₀_constrained.free_to_prescribed_mat)
    free_to_prescribed_offset_t₀ = x₀_constrained.free_to_prescribed_offset
    Nₜ = N_t(inner_gmrf)
    t_idcs = 0:(Nₜ-1)
    Nₛ = N_spatial(inner_gmrf)
    prescribed_dofs = vcat([t_idx * Nₛ .+ prescribed_dofs_t₀ for t_idx in t_idcs]...)
    free_dofs = vcat([t_idx * Nₛ .+ free_dofs_t₀ for t_idx in t_idcs]...)
    free_to_prescribed_mat = kron(sparse(I, (Nₜ, Nₜ)), free_to_prescribed_mat_t₀)
    free_to_prescribed_offset = repeat(free_to_prescribed_offset_t₀, Nₜ)

    return ConstrainedGMRF(
        inner_gmrf,
        prescribed_dofs,
        free_dofs,
        free_to_prescribed_mat,
        free_to_prescribed_offset,
    )
end

CONST_ST_TYPE = Union{
    ConstrainedGMRF{<:ConstantMeshSTGMRF},
    ConstrainedGMRF{<:LinearConditionalGMRF{<:ConstantMeshSTGMRF}},
}

_constrained_st_transform(d::CONST_ST_TYPE, x::AbstractVector) =
    make_chunks(transform_free_to_full(d, x), N_t(d))
time_means(d::CONST_ST_TYPE) = _constrained_st_transform(d, mean(d))
time_vars(d::CONST_ST_TYPE) = _constrained_st_transform(d, var(d))
time_stds(d::CONST_ST_TYPE) = _constrained_st_transform(d, std(d))
time_rands(d::CONST_ST_TYPE, rng::AbstractRNG) =
    _constrained_st_transform(d, rand(rng, d.inner_gmrf))

discretization_at_time(d::CONST_ST_TYPE, t::Int) = discretization_at_time(d.inner_gmrf, t)
N_spatial(d::CONST_ST_TYPE) = N_spatial(d.inner_gmrf)
N_t(d::CONST_ST_TYPE) = N_t(d.inner_gmrf)
