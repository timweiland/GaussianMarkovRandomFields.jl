module GaussianMarkovRandomFields

include("typedefs.jl")
include("utils/utils.jl")
include("linear_maps/linear_maps.jl")
include("preconditioners/preconditioners.jl")
include("gmrf.jl")
include("chordal_gmrf.jl")
include("metagmrf.jl")
include("solvers/solvers.jl")
include("workspace/workspace.jl")
include("autoregressive/autoregressive.jl")
include("spdes/spdes.jl")
include("latent_models/latent_models.jl")
include("fem/fem.jl")
include("observation_models/observation_models.jl")
include("arithmetic/arithmetic.jl")
include("linear_predictor_marginals.jl")
include("formula/constructors.jl")
include("plots/makie.jl")
include("kl_cholesky/base.jl")
include("graphical_lasso/graphical_lasso.jl")
include("workspace/gaussian_approximation.jl")
# High-level latent-prior entry points for gaussian_approximation. Loaded here
# (not from latent_models/) because it dispatches on `ObservationLikelihood` and
# drives both the arithmetic and workspace Newton loops defined above.
include("latent_models/gaussian_approximation.jl")
# AD-defined (TMB-style) latent prior. Loaded after observation_models because it
# reuses the AutoDiffLikelihood machinery (_ADPrepCache, default AD backends, the
# eltype helpers) defined there.
include("latent_models/autodiff_latent_prior.jl")
include("latent_models/structured_latent_prior.jl")
include("observation_models/structured_observation_model.jl")
include("workspace/latent_model_integration.jl")
include("workspace/workspace_pool.jl")
include("autodiff/autodiff.jl")
include("workspace/autodiff.jl")

end
