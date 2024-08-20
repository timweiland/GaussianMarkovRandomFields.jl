export AbstractSpatiotemporalGMRF,
    ConstantMeshSTGMRF,
    time_means,
    time_vars,
    time_stds,
    time_rands,
    discretization_at_time,
    spatial_to_spatiotemporal

################################################################################
#
#    AbstractSpatiotemporalGMRF
#
#    Abstract type for spatiotemporal Gaussian Markov Random Fields.
#    Each subtype must implement the following methods:
#    - N_t
#    - time_means
#    - time_vars
#    - time_stds
#    - time_rands
#    - discretization_at_time
#
################################################################################
"""
    AbstractSpatiotemporalGMRF

A spatiotemporal GMRF is a GMRF that explicitly encodes the spatial and temporal
structure of the underlying random field.
All time points are modelled in one joint GMRF.
It provides utilities to get statistics, draw samples and get the spatial discretization
at a given time.
"""
abstract type AbstractSpatiotemporalGMRF <: AbstractGMRF end

length(x::AbstractSpatiotemporalGMRF) = length(x.mean)
mean(x::AbstractSpatiotemporalGMRF) = x.mean
precision_mat(x::AbstractSpatiotemporalGMRF) = x.precision
precision_chol_precomputed(x::AbstractSpatiotemporalGMRF) = x.precision_chol_precomp
@memoize function precision_chol(x::AbstractSpatiotemporalGMRF)
    if precision_chol_precomputed(x) === nothing
        return cholesky(precision_mat(x))
    end
    return precision_chol_precomputed(x)
end

N_t(::AbstractSpatiotemporalGMRF) = error("N_t not implemented")
time_means(::AbstractSpatiotemporalGMRF) = error("time_means not implemented")
time_vars(::AbstractSpatiotemporalGMRF) = error("time_vars not implemented")
time_stds(::AbstractSpatiotemporalGMRF) = error("time_stds not implemented")
time_rands(::AbstractSpatiotemporalGMRF, ::AbstractRNG) =
    error("time_rands not implemented")
discretization_at_time(::AbstractSpatiotemporalGMRF, ::Int) =
    error("discretization_at_time not implemented")


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
struct ConstantMeshSTGMRF{T} <: AbstractSpatiotemporalGMRF
    mean::AbstractVector{T}
    precision::AbstractMatrix{T}
    discretization::FEMDiscretization
    precision_chol_precomp::Union{Nothing,Cholesky,SparseArrays.CHOLMOD.Factor}

    function ConstantMeshSTGMRF(
        mean::AbstractVector{T},
        precision::AbstractMatrix{T},
        discretization::FEMDiscretization,
        precision_chol::Union{Nothing,Cholesky,SparseArrays.CHOLMOD.Factor} = nothing,
    ) where {T}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        (n % ndofs(discretization)) == 0 || throw(ArgumentError("size mismatch"))
        new{T}(mean, precision, discretization, precision_chol)
    end
end

N_spatial(x::ConstantMeshSTGMRF) = ndofs(x.discretization)
N_t(x::ConstantMeshSTGMRF) = length(x) ÷ N_spatial(x)

@views function make_chunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n - 1 ? end : c * k + c)] for k = 0:n-1]
end

time_means(x::ConstantMeshSTGMRF) = make_chunks(mean(x), N_t(x))
time_vars(x::ConstantMeshSTGMRF) = make_chunks(var(x), N_t(x))
time_stds(x::ConstantMeshSTGMRF) = make_chunks(std(x), N_t(x))
time_rands(x::ConstantMeshSTGMRF, rng::AbstractRNG) = make_chunks(rand(rng, x), N_t(x))
discretization_at_time(x::ConstantMeshSTGMRF, ::Int) = x.discretization

################################################################################
#
# Utils
#
################################################################################

"""
    spatial_to_spatiotemporal(spatial_matrix, t_idx, N_t)

Make a spatial matrix applicable to a spatiotemporal system at time index `t_idx`.
Results in a matrix that selects the spatial information exactly at time `t_idx`.
"""
function spatial_to_spatiotemporal(spatial_matrix::AbstractMatrix, t_idx, N_t)
    E_t = spzeros(1, N_t)
    E_t[t_idx] = 1
    return kron(E_t, spatial_matrix)
end
