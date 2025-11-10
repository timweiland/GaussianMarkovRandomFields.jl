export PoissonObservations, counts, exposure

"""
    PoissonObservations <: AbstractVector{Tuple{Int, Float64}}

Combined observation type for Poisson data containing both counts and exposure.

This type packages Poisson observation data (event counts and exposure, e.g.
time-at-risk, population, or area) into a single vector-like object where
each element is a (count, exposure) tuple.

The intended model is

    yᵢ ∼ Poisson(λᵢ),    λᵢ = exposureᵢ * rateᵢ

so that on the log scale you typically use

    log(λᵢ) = ηᵢ + log(exposureᵢ)

where ηᵢ comes from the latent Gaussian field / linear predictor.

# Fields
- `counts::Vector{Int}`: Number of events for each observation
- `exposure::Vector{Float64}`: Exposure for each observation (e.g. time, area, population)

# Example
```julia
# Create Poisson observations with exposure (e.g. person-years)
y = PoissonObservations([3, 1, 4], [1.0, 2.5, 0.75])

# Access as tuples
y[1]  # (3, 1.0)
y[2]  # (1, 2.5)
"""
struct PoissonObservations <: AbstractVector{Tuple{Int, Float64}}
    counts::Vector{Int}
    logexposure::Vector{Float64}

    # Main constructor: counts + exposure (on natural scale)
    function PoissonObservations(
            counts::AbstractVector{<:Integer},
            exposure::AbstractVector{<:Real}
        )
        counts_int = Int.(counts)
        exposure_f64 = Float64.(exposure)

        if length(counts_int) != length(exposure_f64)
            error("Length of counts ($(length(counts_int))) must match length of exposure ($(length(exposure_f64)))")
        end

        logexposure = similar(exposure_f64)
        for i in eachindex(counts_int)
            if counts_int[i] < 0
                error("Counts must be non-negative at index $i (got $(counts_int[i]))")
            end
            if exposure_f64[i] <= 0
                error("Exposure must be positive at index $i (got $(exposure_f64[i]))")
            end
            logexposure[i] = log(exposure_f64[i])
        end

        return new(counts_int, logexposure)
    end

    # Convenience constructor: counts only → exposure = 1.0 for all
    function PoissonObservations(counts::AbstractVector{<:Integer})
        counts_int = Int.(counts)
        logexposure = zeros(Float64, length(counts_int))  # log(1.0) = 0.0
        for i in eachindex(counts_int)
            if counts_int[i] < 0
                error("Counts must be non-negative at index $i (got $(counts_int[i]))")
            end
        end
        return new(counts_int, logexposure)
    end

end

# AbstractVector interface implementation
Base.size(y::PoissonObservations) = size(y.counts)
Base.getindex(y::PoissonObservations, i::Int) = (y.counts[i], exp(y.logexposure[i]))
Base.getindex(y::PoissonObservations, I) = [y[i] for i in I]
Base.IndexStyle(::Type{PoissonObservations}) = IndexLinear()

# Iteration interface
Base.iterate(y::PoissonObservations, i::Int = 1) = i > length(y) ? nothing : (y[i], i + 1)

# Convenience accessors
"""
    counts(y::PoissonObservations) -> Vector{Int}

Extract the counts vector from Poisson observations.
"""
counts(y::PoissonObservations) = y.counts

"""
    exposure(y::PoissonObservations) -> Vector{Float64}

Extract the exposure vector from Poisson observations.
"""
exposure(y::PoissonObservations) = exp.(y.logexposure)

"""
    logexposure(y::PoissonObservations) -> Vector{Float64}

Extract the log-exposure vector from Poisson observations.
"""
logexposure(y::PoissonObservations) = y.logexposure

function Base.getproperty(y::PoissonObservations, name::Symbol)
    if name === :exposure
        return exp.(getfield(y, :logexposure))
    else
        return getfield(y, name)
    end
end

# COV_EXCL_START

function Base.show(io::IO, y::PoissonObservations)
    return print(io, "PoissonObservations($(length(y)) observations)")
end

function Base.show(io::IO, ::MIME"text/plain", y::PoissonObservations)
    println(io, "$(length(y))-element PoissonObservations:")
    for i in eachindex(y.counts)
        expi = exp(y.logexposure[i])
        println(io, " [$i]: $(y.counts[i]) events over exposure $(expi)")
    end
    return
end

# COV_EXCL_STOP
