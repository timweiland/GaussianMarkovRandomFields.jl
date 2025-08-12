import Base: length
import Distributions:
    AbstractMvNormal,
    mean,
    cov,
    invcov,
    logdetcov,
    sqmahal,
    sqmahal!,
    gradlogpdf,
    _rand!,
    var,
    std
using LinearAlgebra
using Random
using SparseArrays
using LinearMaps
using LinearSolve

"""
    symmetrize(A)

Apply appropriate symmetric wrapper to matrix types.
- `Diagonal`: Return as-is (no wrapping needed)
- `Tridiagonal`: Convert to `SymTridiagonal`
- Others: Wrap in `Symmetric`
"""
symmetrize(A::Diagonal) = A
symmetrize(A::Tridiagonal) = SymTridiagonal(A)  
symmetrize(A::AbstractMatrix) = Symmetric(A)

export AbstractGMRF, GMRF, precision_map, precision_matrix, InformationVector, information_vector

########################################################
#
#    InformationVector
#
#    Wrapper type to distinguish information vector from mean
#    in GMRF constructors
#
########################################################

"""
    InformationVector(data::AbstractVector)

Wrapper type for information vectors (Q * μ) used in GMRF construction.
This allows distinguishing between constructors that take mean vectors 
vs information vectors.
"""
struct InformationVector{T}
    data::Vector{T}
    
    InformationVector(data::AbstractVector{T}) where T = new{T}(Vector{T}(data))
end

Base.length(iv::InformationVector) = length(iv.data)
Base.eltype(::InformationVector{T}) where T = T

########################################################
#
#    AbstractGMRF
#
#    Abstract type for Gaussian Markov Random Fields
#    Each subtype must implement the following methods:
#    - length
#    - mean
#    - precision_map
#
########################################################
"""
    AbstractGMRF

A [Gaussian Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field#Gaussian) 
(GMRF) is a special case of a multivariate normal distribution where the precision matrix
is sparse. The zero entries in the precision correspond to conditional independencies.
"""
abstract type AbstractGMRF{T<:Real, L<:Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}} <: AbstractMvNormal end

linsolve_cache(x::AbstractGMRF) = x.linsolve_cache
mean(s::AbstractGMRF) = s.mean

"""
    precision_map(::AbstractGMRF)

Return the precision (inverse covariance) map of the GMRF.
"""
precision_map(::AbstractGMRF) = error("precision_map not implemented for GMRF")

"""
    precision_matrix(::AbstractGMRF)

Return the precision (inverse covariance) matrix of the GMRF.
"""
precision_matrix(x::AbstractGMRF{T, <:LinearMaps.LinearMap}) where T = to_matrix(precision_map(x))
precision_matrix(x::AbstractGMRF{T, <:AbstractMatrix}) where T = precision_map(x)

length(d::AbstractGMRF) = Base.size(precision_map(d), 1)

### Generic derived methods
invcov(d::AbstractGMRF) = Symmetric(precision_matrix(d))
cov(::AbstractGMRF) = error("Prevented forming dense covariance matrix in memory.")

logdetcov(d::AbstractGMRF) = error("logdetcov not implemented for $(typeof(d))")

sqmahal(d::AbstractGMRF, x::AbstractVector) = (Δ = x - mean(d);
dot(Δ, precision_map(d) * Δ))
sqmahal!(r::AbstractVector, d::AbstractGMRF, x::AbstractVector) = (r .= sqmahal(d, x))

gradlogpdf(d::AbstractGMRF, x::AbstractVector) = -precision_map(d) * (x .- mean(d))

_rand!(rng::AbstractRNG, d::AbstractGMRF, x::AbstractVector) =
    error("_rand! not implemented for $(typeof(d))")

var(d::AbstractGMRF) = error("var not implemented for $(typeof(d))")
std(d::AbstractGMRF) = sqrt.(var(d))

#####################
#
#    GMRF
#
#####################
"""
    GMRF(mean, precision, alg=LinearSolve.DefaultLinearSolver(); Q_sqrt=nothing, rbmc_strategy=RBMCStrategy(1000))

A Gaussian Markov Random Field with mean `mean` and precision matrix `precision`.

# Arguments
- `mean::AbstractVector`: The mean vector of the GMRF.
- `precision::Union{LinearMap, AbstractMatrix}`: The precision matrix (inverse covariance) of the GMRF.
- `alg`: LinearSolve algorithm to use for linear system solving. Defaults to `LinearSolve.DefaultLinearSolver()`.
- `Q_sqrt::Union{Nothing, AbstractMatrix}`: Square root of precision matrix Q, used for sampling when algorithm doesn't support backward solve.
- `rbmc_strategy`: RBMC strategy for marginal variance computation when selected inversion is unavailable. Defaults to `RBMCStrategy(1000)`.

# Type Parameters
- `T<:Real`: The numeric type (e.g., Float64).
- `PrecisionMap<:Union{LinearMap{T}, AbstractMatrix{T}}`: The type of the precision matrix.

# Fields
- `mean::Vector{T}`: The mean vector.
- `precision::PrecisionMap`: The precision matrix.
- `Q_sqrt::Union{Nothing, AbstractMatrix{T}}`: Square root of precision matrix for sampling.
- `linsolve_cache::LinearSolve.LinearCache`: The LinearSolve cache for efficient operations.
- `rbmc_strategy`: RBMC strategy for variance computation fallback.

# Notes
The LinearSolve cache is constructed automatically and is used to compute means, variances, 
samples, and other GMRF quantities efficiently. The algorithm choice determines which 
optimization strategies (selected inversion, backward solve) are available. When selected
inversion is not supported, marginal variances are computed using the configured RBMC strategy.
"""
struct GMRF{T<:Real, PrecisionMap<:Union{LinearMap{T}, AbstractMatrix{T}}, QSqrt, Cache<:LinearSolve.LinearCache, RBMCStrat} <: AbstractGMRF{T, PrecisionMap}
    mean::Vector{T}
    information_vector::Union{Nothing, Vector{T}}
    precision::PrecisionMap
    Q_sqrt::QSqrt
    linsolve_cache::Cache
    rbmc_strategy::RBMCStrat

    # Constructor 1: From mean vector
    function GMRF(
        mean::AbstractVector,
        precision::PrecisionMap,
        alg = nothing; 
        Q_sqrt = nothing,
        rbmc_strategy = RBMCStrategy(1000)
    ) where {PrecisionMap <: Union{LinearMap, AbstractMatrix}}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        T = promote_type(eltype(mean), eltype(precision))
        if eltype(mean) != T
            mean = convert(AbstractVector{T}, mean)
        end
        if eltype(precision) != T
            if precision isa LinearMap
                precision = LinearMap{T}(convert(AbstractMatrix{T}, to_matrix(precision)))
            else
                precision = convert(AbstractMatrix{T}, precision)
            end
        end

        # Set up LinearSolve cache
        # For LinearMaps, we need to convert to matrix for LinearSolve
        precision_matrix = precision isa LinearMap ? to_matrix(precision) : precision
        # Use appropriate symmetric wrapper for different matrix types
        prob = LinearProblem(symmetrize(precision_matrix), copy(mean))
        linsolve_cache = init(prob, alg)
        
        return new{T, typeof(precision), typeof(Q_sqrt), typeof(linsolve_cache), typeof(rbmc_strategy)}(mean, nothing, precision, Q_sqrt, linsolve_cache, rbmc_strategy)
    end

    # Constructor 2: From information vector
    function GMRF(
        information::InformationVector{T},
        precision::PrecisionMap,
        alg = nothing; 
        Q_sqrt = nothing,
        rbmc_strategy = RBMCStrategy(1000)
    ) where {T<:Real, PrecisionMap <: Union{LinearMap{T}, AbstractMatrix{T}}}
        n = length(information)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))

        # Set up LinearSolve cache and solve for mean
        precision_matrix = precision isa LinearMap ? to_matrix(precision) : precision
        prob = LinearProblem(symmetrize(precision_matrix), copy(information.data))
        linsolve_cache = init(prob, alg)
        mean = solve!(linsolve_cache).u
        
        return new{T, typeof(precision), typeof(Q_sqrt), typeof(linsolve_cache), typeof(rbmc_strategy)}(mean, information.data, precision, Q_sqrt, linsolve_cache, rbmc_strategy)
    end

end

length(d::GMRF) = length(d.mean)
mean(d::GMRF) = d.mean
precision_map(d::GMRF) = d.precision

function Base.show(io::IO, d::GMRF{T}) where T
    print(io, "GMRF{$T}(n=$(length(d)), alg=$(typeof(d.linsolve_cache.alg)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::GMRF{T}) where T
    println(io, "GMRF{$T} with $(length(d)) variables")
    println(io, "  Algorithm: $(typeof(d.linsolve_cache.alg))")
    println(io, "  Mean: $(mean(d))")
    if d.Q_sqrt !== nothing
        print(io, "  Q_sqrt: available")
    else
        print(io, "  Q_sqrt: not available")
    end
end

"""
    information_vector(d::GMRF)

Return the information vector (Q * μ) for the GMRF.
If stored, returns the cached value; otherwise computes it.
"""
function information_vector(d::GMRF)
    if d.information_vector !== nothing
        return d.information_vector
    else
        return precision_map(d) * mean(d)
    end
end

# Implement core GMRF operations using LinearSolve cache
function logdetcov(d::GMRF)
    return logdet_cov(d.linsolve_cache)
end

function _rand!(rng::AbstractRNG, d::GMRF, x::AbstractVector)
    return _rand_impl!(rng, d, x, supports_backward_solve(d.linsolve_cache.alg))
end

function _rand_impl!(rng::AbstractRNG, d::GMRF, x::AbstractVector, ::Val{true})
    # Use backward solve
    randn!(rng, x)
    x .= backward_solve(d.linsolve_cache, x)
    x .+= d.mean
    return x
end

function _rand_impl!(rng::AbstractRNG, d::GMRF, x::AbstractVector, ::Val{false})
    # Fallback to Q_sqrt approach
    if d.Q_sqrt === nothing
        error("Cannot sample from GMRF: algorithm $(typeof(d.linsolve_cache.alg)) doesn't support backward solve and Q_sqrt is nothing")
    end
    # Sample z ~ N(0,I), compute w = √Q * z, solve Q * x = w
    z = randn(rng, size(d.Q_sqrt, 2))
    w = d.Q_sqrt * z
    # Update RHS and solve
    d.linsolve_cache.b .= w
    solve!(d.linsolve_cache)
    x .= d.linsolve_cache.u .+ d.mean
    return x
end

function var(d::GMRF)
    return _var_impl(d, supports_selinv(d.linsolve_cache.alg))
end

function _var_impl(d::GMRF, ::Val{true})
    # Use selected inversion
    return selinv_diag(d.linsolve_cache)
end

function _var_impl(d::GMRF, ::Val{false})
    # Fallback to RBMC
    return var(d, d.rbmc_strategy)
end

