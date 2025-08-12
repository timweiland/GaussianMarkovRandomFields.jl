using LinearMaps, LinearAlgebra, SparseArrays

export ConstantMeshSTGMRF, ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF,
       ImplicitEulerMetadata, ConcreteSTMetadata

################################################################################
#
#    ConstantMeshSTGMRF - MetaGMRF-based design
#
#    Spatiotemporal GMRF metadata types and constructors
#
################################################################################

"""
    ImplicitEulerMetadata{D, SSM} <: GMRFMetadata

Metadata for implicit Euler spatiotemporal GMRFs with constant spatial mesh.
"""
struct ImplicitEulerMetadata{D, SSM} <: GMRFMetadata
    discretization::FEMDiscretization{D}
    ssm::SSM
    N_spatial::Int
    N_t::Int
end

"""
    ConcreteSTMetadata{D} <: GMRFMetadata

Metadata for concrete spatiotemporal GMRFs with constant spatial mesh.
"""
struct ConcreteSTMetadata{D} <: GMRFMetadata
    discretization::FEMDiscretization{D}
    N_spatial::Int
    N_t::Int
end

"""
    ImplicitEulerConstantMeshSTGMRF

A spatiotemporal GMRF with constant spatial discretization and an implicit Euler
discretization of the temporal dynamics. Uses MetaGMRF for clean type structure.
"""
const ImplicitEulerConstantMeshSTGMRF{D, SSM, T, P, G} = MetaGMRF{ImplicitEulerMetadata{D, SSM}, T, P, G}

"""
    ConcreteConstantMeshSTGMRF

A concrete implementation of a spatiotemporal GMRF with constant spatial
discretization. Uses MetaGMRF for clean type structure.
"""
const ConcreteConstantMeshSTGMRF{D, T, P, G} = MetaGMRF{ConcreteSTMetadata{D}, T, P, G}

# Type aliases for backward compatibility
const ConstantMeshSTGMRF = Union{ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF}

# Constructors
function ImplicitEulerConstantMeshSTGMRF(
    gmrf::AbstractGMRF,
    discretization::FEMDiscretization{D},
    ssm::SSM,
) where {D, SSM}
    n = length(gmrf)
    (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))
    N_spatial = ndofs(discretization)
    N_t = n ÷ N_spatial
    
    metadata = ImplicitEulerMetadata(discretization, ssm, N_spatial, N_t)
    return MetaGMRF(gmrf, metadata)
end

function ConcreteConstantMeshSTGMRF(
    gmrf::AbstractGMRF,
    discretization::FEMDiscretization{D},
) where {D}
    n = length(gmrf)
    (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))
    N_spatial = ndofs(discretization)
    N_t = n ÷ N_spatial
    
    metadata = ConcreteSTMetadata(discretization, N_spatial, N_t)
    return MetaGMRF(gmrf, metadata)
end

# Spatiotemporal-specific operations that access metadata
N_spatial(x::ConstantMeshSTGMRF) = x.metadata.N_spatial
N_t(x::ConstantMeshSTGMRF) = x.metadata.N_t

time_means(x::ConstantMeshSTGMRF) = make_chunks(mean(x), N_t(x))
time_vars(x::ConstantMeshSTGMRF) = make_chunks(var(x), N_t(x))
time_stds(x::ConstantMeshSTGMRF) = make_chunks(std(x), N_t(x))
time_rands(x::ConstantMeshSTGMRF, rng::AbstractRNG) = make_chunks(rand(rng, x), N_t(x))
discretization_at_time(x::ConstantMeshSTGMRF, ::Int) = x.metadata.discretization

# Access SSM for ImplicitEuler types
ssm(x::ImplicitEulerConstantMeshSTGMRF) = x.metadata.ssm

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
