export NegativeBinomialObservations

"""
    NegativeBinomialObservations <: AbstractVector{Tuple{Int, Float64}}

Combined observation type for Negative Binomial data containing counts and exposure.

The intended model is

    yᵢ ∼ NegativeBinomial(r, pᵢ),    pᵢ = r / (r + μᵢ),    μᵢ = exposureᵢ * rateᵢ

so that on the log scale you typically use

    log(μᵢ) = ηᵢ + log(exposureᵢ)

where ηᵢ comes from the latent Gaussian field / linear predictor.

# Fields
- `counts::Vector{Int}`: Number of events for each observation
- `logexposure::Vector{Float64}`: Log-exposure for each observation

# Example
```julia
# With exposure
y = NegativeBinomialObservations([3, 1, 8], [1.0, 2.5, 0.75])

# Without exposure (exposure = 1.0)
y = NegativeBinomialObservations([3, 1, 8])
```
"""
struct NegativeBinomialObservations <: AbstractVector{Tuple{Int, Float64}}
    counts::Vector{Int}
    logexposure::Vector{Float64}

    function NegativeBinomialObservations(
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

    function NegativeBinomialObservations(counts::AbstractVector{<:Integer})
        counts_int = Int.(counts)
        logexposure = zeros(Float64, length(counts_int))
        for i in eachindex(counts_int)
            if counts_int[i] < 0
                error("Counts must be non-negative at index $i (got $(counts_int[i]))")
            end
        end
        return new(counts_int, logexposure)
    end
end

Base.size(y::NegativeBinomialObservations) = size(y.counts)
Base.getindex(y::NegativeBinomialObservations, i::Int) = (y.counts[i], exp(y.logexposure[i]))
Base.getindex(y::NegativeBinomialObservations, I) = [y[i] for i in I]
Base.IndexStyle(::Type{NegativeBinomialObservations}) = IndexLinear()
Base.iterate(y::NegativeBinomialObservations, i::Int = 1) = i > length(y) ? nothing : (y[i], i + 1)

"""
    counts(y::NegativeBinomialObservations) -> Vector{Int}

Extract the counts vector from Negative Binomial observations.
"""
counts(y::NegativeBinomialObservations) = y.counts

"""
    exposure(y::NegativeBinomialObservations) -> Vector{Float64}

Extract the exposure vector from Negative Binomial observations.
"""
exposure(y::NegativeBinomialObservations) = exp.(y.logexposure)

"""
    logexposure(y::NegativeBinomialObservations) -> Vector{Float64}

Extract the log-exposure vector from Negative Binomial observations.
"""
logexposure(y::NegativeBinomialObservations) = y.logexposure

function Base.getproperty(y::NegativeBinomialObservations, name::Symbol)
    if name === :exposure
        return exp.(getfield(y, :logexposure))
    else
        return getfield(y, name)
    end
end

# COV_EXCL_START

function Base.show(io::IO, y::NegativeBinomialObservations)
    return print(io, "NegativeBinomialObservations($(length(y)) observations)")
end

function Base.show(io::IO, ::MIME"text/plain", y::NegativeBinomialObservations)
    println(io, "$(length(y))-element NegativeBinomialObservations:")
    for i in eachindex(y.counts)
        expi = exp(y.logexposure[i])
        println(io, " [$i]: $(y.counts[i]) events over exposure $(expi)")
    end
    return
end

# COV_EXCL_STOP
