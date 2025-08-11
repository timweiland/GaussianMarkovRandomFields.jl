using LinearMaps, LinearAlgebra, SparseArrays

export ConstantMeshSTGMRF, ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF

################################################################################
#
#    ConstantMeshSTGMRF - New forwarding design
#
#    Spatiotemporal GMRF types that forward operations to underlying GMRF
#
################################################################################

"""
    ImplicitEulerConstantMeshSTGMRF{D}

A spatiotemporal GMRF with constant spatial discretization and an implicit Euler
discretization of the temporal dynamics. Forwards all GMRF operations to the underlying GMRF.
"""
struct ImplicitEulerConstantMeshSTGMRF{D, T, PrecisionMap, QSqrt, Cache}
    gmrf::GMRF{T, PrecisionMap, QSqrt, Cache}
    discretization::FEMDiscretization{D}
    ssm::ImplicitEulerSSM
    N_spatial::Int
    N_t::Int

    function ImplicitEulerConstantMeshSTGMRF(
        gmrf::GMRF{T, PrecisionMap, QSqrt, Cache},
        discretization::FEMDiscretization{D},
        ssm::ImplicitEulerSSM,
    ) where {D, T, PrecisionMap, QSqrt, Cache}
        n = length(gmrf)
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))
        N_spatial = ndofs(discretization)
        N_t = n ÷ N_spatial
        
        return new{D, T, PrecisionMap, QSqrt, Cache}(
            gmrf, discretization, ssm, N_spatial, N_t
        )
    end
end

"""
    ConcreteConstantMeshSTGMRF{D}

A concrete implementation of a spatiotemporal GMRF with constant spatial
discretization. Forwards all GMRF operations to the underlying GMRF.
"""
struct ConcreteConstantMeshSTGMRF{D, T, PrecisionMap, QSqrt, Cache}
    gmrf::GMRF{T, PrecisionMap, QSqrt, Cache}
    discretization::FEMDiscretization{D}
    N_spatial::Int
    N_t::Int

    function ConcreteConstantMeshSTGMRF(
        gmrf::GMRF{T, PrecisionMap, QSqrt, Cache},
        discretization::FEMDiscretization{D},
    ) where {D, T, PrecisionMap, QSqrt, Cache}
        n = length(gmrf)
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))
        N_spatial = ndofs(discretization)
        N_t = n ÷ N_spatial
        
        return new{D, T, PrecisionMap, QSqrt, Cache}(
            gmrf, discretization, N_spatial, N_t
        )
    end
end

# Forward all core GMRF operations to the underlying GMRF
@forward ImplicitEulerConstantMeshSTGMRF.gmrf (
    length, mean, precision_map, var, std, rand, information_vector
)
@forward ConcreteConstantMeshSTGMRF.gmrf (
    length, mean, precision_map, var, std, rand, information_vector
)

# Type aliases for backward compatibility
const ConstantMeshSTGMRF = Union{ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF}

# Spatiotemporal-specific operations that need the metadata
N_spatial(x::ConstantMeshSTGMRF) = x.N_spatial
N_t(x::ConstantMeshSTGMRF) = x.N_t

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
# LinearConditionalGMRF methods temporarily removed - will need updating for new linear_condition approach
# N_spatial(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = ndofs(x.prior.discretization)
# N_t(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = length(x) ÷ N_spatial(x)

# time_means(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(mean(x), N_t(x))
# time_vars(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(var(x), N_t(x))
# time_stds(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}) = make_chunks(std(x), N_t(x))
# time_rands(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}, rng::AbstractRNG) =
#     make_chunks(rand(rng, x), N_t(x))
# discretization_at_time(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF}, ::Int) =
#     x.prior.discretization


function default_preconditioner_strategy(
    x::ConstantMeshSTGMRF,
)
    block_size = N_spatial(x)
    Q = sparse(to_matrix(precision_map(x)))
    return temporal_block_gauss_seidel(Q, block_size)
end
