# # Advanced GMRF modelling for disease mapping
#
# In this walk‑through we fit a simple, interpretable disease‑mapping model to the
# classic Scotland lip cancer data. The response is the number of observed cases
# per district. We use a Poisson likelihood with an exposure offset and a BYM
# (Besag–York–Mollié) latent field to capture spatial structure and local
# over‑dispersion. We also include a fixed‑effect covariate.
#
# What we put together:
# - A Poisson model with a log link and log‑exposure offset (so we model relative risk).
# - A BYM latent field = Besag (structured spatial) + IID (unstructured spatial).
# - Fixed effects (intercept + one covariate).
#
# What you’ll learn:
# - Build a polygon contiguity adjacency matrix directly from a shapefile.
# - Compose a BYM + fixed‑effects model via the formula interface.
# - Run a fast Gaussian approximation and make useful maps.
# - Read fixed effects, split BYM into structured/unstructured, and add an
#   exceedance map (P(RR > 1)).

using CSV, DataFrames
using StatsModels
using SparseArrays
using Distributions
using GaussianMarkovRandomFields
using Shapefile              # Activates the Shapefile extension for contiguity
using LibGEOS
using Plots
using GeoInterface
using Random, Statistics

# ## Load data
# We load from a local `data/` directory next to this tutorial. The CSV has observed counts (CANCER),
# expected counts (CEXP), and the covariate AFF: the proportion of the population
# engaged in Agriculture, Fishing or Forestry (hence the name). The shapefile
# contains district polygons with a `CODE` attribute. We align the CSV rows to
# the shapefile feature order and keep `CODE` as a convenient string identifier.

data_dir = joinpath(@__DIR__, "data"); mkpath(data_dir)

base_url = "https://github.com/timweiland/GaussianMarkovRandomFields.jl/raw/refs/heads/main/docs/data"
extensions = ["csv", "shp", "dbf"]
paths = [joinpath(data_dir, "scotlip.$(ext)") for ext in extensions]
csv_path, shp_path, _ = paths
for (path, extension) in zip(paths, extensions)
    if !isfile(path)
        try
            download("$base_url/scotlip.$(extension)", path)
        catch err
            error("Could not download scotlip.$(extension); place it at $(path).")
        end
    end
end

#
df = DataFrame(CSV.File(csv_path))
rename!(df, Dict(:CANCER => :y, :CEXP => :E, :AFF => :aff, :CODE => :code, :NAME => :name))
df.offset = log.(df.E)
df.aff ./= 100.0 # AFF is provided in percent; convert to fraction for stable scaling
df[1:5, :]

# Build adjacency matrix from the shapefile and obtain feature IDs (by CODE)
W, ids = contiguity_adjacency(shp_path; id_field = :CODE)
id_to_node = Dict(ids[i] => i for i in eachindex(ids))
W

# ## Model in one picture
# The (log) relative risk per district is modelled as
#
#   log RRᵢ = β₀ + β_aff · AFFᵢ + uᵢ (Besag) + vᵢ (IID)
#
# and counts as yᵢ ~ Poisson(Eᵢ · RRᵢ), where Eᵢ is the expected count (offset).
#
# In code, we specify:
# - `besag = Besag(W; id_to_node = Dict(code => i))` for the structured effect.
# - `IID(code)` for the unstructured effect.
# - `1 + aff` for fixed effects (intercept + AFF). The fixed block is weakly
#   regularized internally for numerical stability.

besag = Besag(W; id_to_node = id_to_node, singleton_policy = :degenerate)

f = @formula(y ~ 1 + aff + IID(code) + besag(code))
comp = build_formula_components(f, df; family = Poisson)

# ## Likelihood with offset and Gaussian approximation
# The Poisson observation model uses the log‑link by default. Passing
# `offset = log(E)` encodes exposure and turns the linear predictor into log RR.
lik = comp.obs_model(df.y; offset = df.offset)

# Pick prior precision scales for the BYM parts (you can tune these later) and
# form a Gaussian approximation of the posterior.
prior = comp.combined_model(; τ_besag = 4.1, τ_iid = 3.9)
post = gaussian_approximation(prior, lik)

# Linear predictor on the observation scale (η = log RR) and relative risk
η = comp.A * mean(post)
RR = exp.(η)
(minimum(RR), median(RR), maximum(RR))

# ## Choropleth: polygons filled by relative risk
# This is the “money map”: areas in purple have elevated RR; yellowish areas are
# comparatively lower. Multipolygons are handled by duplicating the area’s value
# across its parts.
table = Shapefile.Table(shp_path)
geoms = GeoInterface.convert.(Ref(LibGEOS), table.geometry)

shape_from_polygon(poly::LibGEOS.Polygon) = begin
    ring = LibGEOS.exteriorRing(poly)
    mp = LibGEOS.uniquePoints(ring)            # MultiPoint of ring vertices
    pts = LibGEOS.getGeometries(mp)            # Vector{Point}
    xs = Float64[LibGEOS.getXMin(p) for p in pts]
    ys = Float64[LibGEOS.getYMin(p) for p in pts]
    Shape(xs, ys)
end

shapes = Shape[]
zvals = Float64[]
@inbounds for (i, g) in enumerate(geoms)
    if g isa LibGEOS.Polygon
        push!(shapes, shape_from_polygon(g))
        push!(zvals, RR[i])
    elseif g isa LibGEOS.MultiPolygon
        for poly in LibGEOS.getGeometries(g)
            push!(shapes, shape_from_polygon(poly))
            push!(zvals, RR[i])
        end
    end
end

plt = plot(
    shapes; fill_z = reshape(zvals, 1, :), st = :shape, c = :viridis, lc = :gray30,
    axis = nothing, aspect_ratio = 1.0, legend = false,
    title = "Relative Risk (BYM + fixed)", cb = :right
)
plt

# ## Fixed effects summary (β and 95% CI)
# Interpreting fixed effects is easy in a log‑link Poisson GLM:
# - exp(β₀) is the baseline relative risk when AFF=0 and random effects are zero.
# - exp(β_aff) is the multiplicative change in RR per unit increase in AFF
#   (here AFF is a fraction; e.g. +0.10 = +10 percentage points). Larger positive
#   β_aff means higher RR in districts with greater AFF.
#
# The code below simply extracts the fixed‑effects block and reports estimates and
# approximate 95% intervals.

n_fixed = comp.meta.n_fixed
n_cols = size(comp.A, 2)
fix_rng = (n_cols - n_fixed + 1):n_cols

β̂_intercept, β̂_aff = mean(post)[fix_rng]
se_intercept, se_aff = std(post)[fix_rng]
z = 1.96  # 95% normal-approx interval
println("Intercept: $(β̂_intercept) ± $(z * se_intercept)")
println("AFF coefficient: $(β̂_aff) ± $(z * se_aff)")

# ## BYM decomposition: structured vs unstructured effects
# Two quick maps help understand “signal vs noise”:
# - Structured (Besag): smooth, neighbor‑sharing spatial pattern.
# - Unstructured (IID): local over‑dispersion not explained by smooth structure.

sizes = comp.combined_model.component_sizes
offs = cumsum([0; sizes[1:(end - 1)]])
idx_block(k) = (offs[k] + 1):(offs[k] + sizes[k])

# Given RHS order here: IID first, Besag second (fixed appended last).
rng_iid = idx_block(1)
rng_besag = idx_block(2)

u_iid = mean(post)[rng_iid]
u_besag = mean(post)[rng_besag]

plt_besag = plot(
    shapes; fill_z = Float64.(u_besag), st = :shape, c = :balance, lc = :gray30,
    legend = false, axis = nothing, aspect_ratio = 1.0,
    title = "Structured effect", cb = :bottom
)
plt_iid = plot(
    shapes; fill_z = Float64.(u_iid), st = :shape, c = :balance, lc = :gray30,
    legend = false, axis = nothing, aspect_ratio = 1.0,
    title = "Unstructured effect", cb = :bottom
)
plot(plt_besag, plt_iid; layout = (1, 2))

# ## Exceedance probability map: P(RR > 1)
# A very actionable view: the probability that RR exceeds 1 (i.e., rates are
# elevated vs expectation). Values near 1 indicate strong evidence for elevation;
# near 0 suggests lower‑than‑expected; around 0.5 is uncertain.
using Random
N = 2000
rng = Random.MersenneTwister(42)
exceed = zeros(length(df.y))
for _ in 1:N
    x = rand(rng, post)
    ηs = comp.A * x
    exceed .+= (ηs .> 0.0)
end
exceed ./= N

plt_ex = plot(
    shapes; fill_z = reshape(Float64.(exceed), 1, :), st = :shape, c = :viridis, lc = :gray30,
    legend = false, axis = nothing, aspect_ratio = 1.0,
    title = "Exceedance P(RR > 1)", cb = :right
)
plt_ex

# ## Notes
# - The `id_to_node` mapping allows using string IDs in the formula; we avoid
#   forcing integer region indices on users.
# - Offsets are supported for Poisson with LogLink; here `offset = log(E)` encodes exposure.
# - Intrinsic Besag imposes sum-to-zero constraints per connected component.
# - Isolated islands (with no edges) receive a degenerate prior, such that only the IID
#   part has an effect there.
# - For more realism, you can tune `τ_besag` and `τ_iid` and add further
#   covariates to the fixed-effects part of the formula.
