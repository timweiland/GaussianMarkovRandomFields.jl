# Script to generate the Bernoulli classification figure for the JOSS paper
# Run from project root: julia --project=paper paper/generate_figure.jl

using DataFrames
using GaussianMarkovRandomFields
## MaternModel and PointEvaluationObsModel live in the FEM extension, which
## activates only once all four of its trigger packages are loaded.
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using LinearAlgebra, Random
using Plots

# Load data
using CodecBzip2, RData
data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)
local_rda = joinpath(data_dir, "lansing_trees.rda")

if !isfile(local_rda)
    repo_url = "https://github.com/spatstat/spatstat.data/raw/refs/heads/master/data/lansing.rda"
    download(repo_url, local_rda)
end

objs = RData.load(local_rda)["lansing"]
x, y_coord = objs[objs.name2index["x"]], objs[objs.name2index["y"]]
is_hickory = objs[objs.name2index["marks"]] .== "hickory"

df = DataFrame(["x" => x, "y" => y_coord, "is_hickory" => is_hickory])
df = unique(df)

X = Matrix(df[:, [:x, :y]])
y = Vector{Int}(df[:, :is_hickory])

# Train/test split
Random.seed!(42)
n = size(X, 1)
perm = randperm(n)
split = round(Int, 0.8n)
train_idcs = perm[1:split]
test_idcs = perm[(split + 1):end]

X_train, y_train = X[train_idcs, :], y[train_idcs]
X_test, y_test = X[test_idcs, :], y[test_idcs]

# Latent model
latent = MaternModel(X; smoothness = 1)
u = latent(τ = 1.0, range = 0.2)

# Bernoulli observations
import Distributions
obs_model = PointEvaluationObsModel(latent.discretization, X_train, Distributions.Bernoulli)
lik = obs_model(y_train)

# Inference
post = gaussian_approximation(u, lik)

# Evaluation
obs_model_test = PointEvaluationObsModel(latent.discretization, X_test, Distributions.Bernoulli)
pred_dist_test = conditional_distribution(obs_model_test, mean(post))
ŷ_prob = mean(pred_dist_test)
ŷ = Int.(ŷ_prob .>= 0.5)
accuracy = sum(ŷ .== y_test) / length(y_test)

# Visualization
xmin, xmax = extrema(X[:, 1])
ymin, ymax = extrema(X[:, 2])
nx, ny = 100, 100
xs = range(xmin, xmax; length = nx)
ys = range(ymin, ymax; length = ny)

grid_points = Array{Float64}(undef, nx * ny, 2)
for (i, (yv, xv)) in enumerate(Iterators.product(ys, xs))
    grid_points[i, 1] = xv
    grid_points[i, 2] = yv
end

obs_grid = PointEvaluationObsModel(latent.discretization, grid_points, Distributions.Bernoulli)
pred_grid = conditional_distribution(obs_grid, mean(post))
probs = reshape(mean(pred_grid), (nx, ny))

# JOSS uses sans-serif fonts; set transparent background, no legend (described in caption)
plt = heatmap(
    xs, ys, probs; xlabel = "x", ylabel = "y",
    title = "P(hickory)",
    colorbar = true, legend = false,
    size = (600, 500),
    dpi = 300,
    fontfamily = "Helvetica",
    background_color = :transparent,
    foreground_color = :black
)

class0 = findall(==(0), y)
class1 = findall(==(1), y)

## Stroke colour matches the fill: the default dark outline dominates markers this
## small and muddies the two classes together in print. Distinct marker shapes give
## a second, colour-independent cue.
scatter!(
    plt, X[class1, 1], X[class1, 2]; m = :circle, ms = 2.5, c = :tomato,
    markerstrokecolor = :tomato, markerstrokewidth = 0.5, label = false
)
scatter!(
    plt, X[class0, 1], X[class0, 2]; m = :diamond, ms = 2.5, c = :royalblue,
    markerstrokecolor = :royalblue, markerstrokewidth = 0.5, label = false
)

# Save to paper/ directory
savefig(plt, joinpath(@__DIR__, "bernoulli_classification.pdf"))
println("Figure saved to paper/bernoulli_classification.pdf")
