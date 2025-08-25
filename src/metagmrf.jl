using LinearAlgebra

export MetaGMRF, GMRFMetadata

"""
    GMRFMetadata

Abstract base type for metadata that can be attached to GMRFs via MetaGMRF.
Concrete subtypes should contain domain-specific information about the GMRF
structure, coordinates, naming, etc.
"""
abstract type GMRFMetadata end

"""
    MetaGMRF{M <: GMRFMetadata, T, P, G <: AbstractGMRF{T, P}} <: AbstractGMRF{T, P}

A wrapper that combines a core GMRF with metadata of type M.
This allows for specialized behavior based on the metadata type
while preserving the computational efficiency of the underlying GMRF.

# Fields
- `gmrf::G`: The core computational GMRF (parametric type)
- `metadata::M`: Domain-specific metadata

# Usage
```julia
# Define metadata types
struct SpatialMetadata <: GMRFMetadata
    coordinates::Matrix{Float64}
    boundary_info::Vector{Int}
end

# Create wrapped GMRF
meta_gmrf = MetaGMRF(my_gmrf, SpatialMetadata(coords, boundary))

# Dispatch on metadata type for specialized behavior
function some_spatial_operation(mgmrf::MetaGMRF{SpatialMetadata})
    # Access coordinates via mgmrf.metadata.coordinates
    # Access GMRF via mgmrf.gmrf
end
```
"""
struct MetaGMRF{M <: GMRFMetadata, T, P, G <: AbstractGMRF{T, P}} <: AbstractGMRF{T, P}
    gmrf::G
    metadata::M

    function MetaGMRF(gmrf::G, metadata::M) where {M <: GMRFMetadata, T, P, G <: AbstractGMRF{T, P}}
        return new{M, T, P, G}(gmrf, metadata)
    end
end

# Forward core GMRF operations to the inner gmrf
Base.length(mgmrf::MetaGMRF) = length(mgmrf.gmrf)
mean(mgmrf::MetaGMRF) = mean(mgmrf.gmrf)
precision_map(mgmrf::MetaGMRF) = precision_map(mgmrf.gmrf)
precision_matrix(mgmrf::MetaGMRF) = precision_matrix(mgmrf.gmrf)
information_vector(mgmrf::MetaGMRF) = information_vector(mgmrf.gmrf)
var(mgmrf::MetaGMRF) = var(mgmrf.gmrf)
std(mgmrf::MetaGMRF) = std(mgmrf.gmrf)
function _rand!(rng::AbstractRNG, mgmrf::MetaGMRF, x::AbstractVector)
    return _rand!(rng, mgmrf.gmrf, x)
end
cov(mgmrf::MetaGMRF) = cov(mgmrf.gmrf)
invcov(mgmrf::MetaGMRF) = invcov(mgmrf.gmrf)
logdetcov(mgmrf::MetaGMRF) = logdetcov(mgmrf.gmrf)
sqmahal(mgmrf::MetaGMRF, x::AbstractVector) = sqmahal(mgmrf.gmrf, x)
sqmahal!(r::AbstractVector, mgmrf::MetaGMRF, x::AbstractVector) = sqmahal!(r, mgmrf.gmrf, x)
gradlogpdf(mgmrf::MetaGMRF, x::AbstractVector) = gradlogpdf(mgmrf.gmrf, x)

# Show methods for better UX
# COV_EXCL_START
function Base.show(io::IO, mgmrf::MetaGMRF{M}) where {M}
    return print(io, "MetaGMRF{", nameof(M), "}(", mgmrf.gmrf, ")")
end

function Base.show(io::IO, ::MIME"text/plain", mgmrf::MetaGMRF{M}) where {M}
    println(io, "MetaGMRF{", nameof(M), "}")
    println(io, "  Inner GMRF: ", mgmrf.gmrf)
    return print(io, "  Metadata: ", mgmrf.metadata)
end
# COV_EXCL_STOP
