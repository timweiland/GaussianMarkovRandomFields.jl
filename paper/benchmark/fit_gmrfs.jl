# GaussianMarkovRandomFields.jl side of the benchmark. For each mesh level written by
# gen_meshes.R: simulate Poisson counts from a Matérn SPDE field (written to y.csv so
# R-INLA fits the same data), then time a Gaussian approximation of the posterior plus all
# latent marginal variances at fixed hyperparameters.
#
#   julia --project=paper/benchmark paper/benchmark/fit_gmrfs.jl <workdir>
#
# Single-threaded (one BLAS thread), to match fit_rinla.R.

using GaussianMarkovRandomFields, Distributions
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using CSV, DataFrames, LinearAlgebra, Random, Statistics

const RANGE = 0.3
const SIGMA = 1.0

function load_discretization(d)
    nodes = CSV.read(joinpath(d, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(d, "triangles.csv"), DataFrame)
    grid = Ferrite.Grid(
        [Ferrite.Triangle((r.v1, r.v2, r.v3)) for r in eachrow(tris)],
        [Ferrite.Node((r.x, r.y)) for r in eachrow(nodes)],
    )
    ip = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    return FEMDiscretization(grid, ip, Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
end

nodes_of(d) = (n = CSV.read(joinpath(d, "nodes.csv"), DataFrame); (n.x, n.y))

# The timed computation: precision at the given hyperparameters, Gaussian approximation
# of the posterior, and all latent marginal variances (via selected inversion).
function fit(latent, obs_model, y)
    prior = latent(τ = 1 / SIGMA^2, range = RANGE)
    post = gaussian_approximation(prior, obs_model(y))
    return mean(post), var(post)
end

function main(workdir)
    BLAS.set_num_threads(1)
    rows = DataFrame()
    for d in sort(filter(startswith("level_"), readdir(workdir)); by = s -> parse(Int, s[7:end]))
        dir = joinpath(workdir, d)
        disc = load_discretization(dir)
        latent = MaternModel(disc; smoothness = 0)  # ν = 1, i.e. α = 2 in 2D
        obs = CSV.read(joinpath(dir, "obs_coords.csv"), DataFrame)
        obs_model = PointEvaluationObsModel(disc, Matrix(hcat(obs.s1, obs.s2)), Poisson)

        Random.seed!(1)
        field = rand(latent(τ = 1 / SIGMA^2, range = RANGE))
        counts = rand.(Poisson.(exp.(evaluation_matrix(disc, Matrix(hcat(obs.s1, obs.s2))) * field)))
        CSV.write(joinpath(dir, "y.csv"), DataFrame(y = counts))
        y = PoissonObservations(counts)

        t_first = @elapsed fit(latent, obs_model, y)  # includes compilation on the first level
        n_rep = length(field) > 20_000 ? 3 : 7
        times = [@elapsed fit(latent, obs_model, y) for _ in 1:n_rep]
        μ, v = fit(latent, obs_model, y)
        # Degrees of freedom are numbered by Ferrite, not in mesh-node order. Evaluating at the
        # nodes gives the permutation, so the summary can be compared with R-INLA's node by node.
        A_nodes = evaluation_matrix(disc, Matrix(hcat(nodes_of(dir)...)))
        perm = [argmax(A_nodes[i, :]) for i in axes(A_nodes, 1)]
        CSV.write(joinpath(dir, "gmrfs_field_summary.csv"), DataFrame(mean = μ[perm], sd = sqrt.(v[perm])))

        push!(
            rows, (;
                level = parse(Int, d[7:end]), n = length(field), m = length(counts),
                t_first, t_median = median(times), t_min = minimum(times),
            )
        )
        println(rows[end, :])
        flush(stdout)
    end
    CSV.write(joinpath(workdir, "gmrfs_times.csv"), rows)
    return rows
end

main(ARGS[1])
