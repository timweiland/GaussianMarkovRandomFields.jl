module GaussianMarkovRandomFields

# --- Dependencies -------------------------------------------------------------
# Packages loaded in full. More targeted imports (`using X: f`, `import X: f`) sit next to the
# code that needs them; like these, they apply to the whole module.
using AMD
using ChainRulesCore
import CliqueTrees
using DataStructures
import DifferentiationInterface as DI
using Distributions
using Graphs
using LinearAlgebra
using LinearMaps
using LinearSolve
using NearestNeighbors
using Random
using SelectedInversion
using SparseArrays
using StatsFuns

# --- Public API ---------------------------------------------------------------

# GMRF types and core operations
export AbstractGMRF, GMRF, ChordalGMRF, MetaGMRF, GMRFMetadata, ConstrainedGMRF,
    InformationVector, information_vector, precision_map, precision_matrix, mean, std,
    joint_gmrf

# Conditioning and Gaussian approximations
export gaussian_approximation, linear_condition, linear_predictor_marginals,
    marginal_loglikelihood

# Latent models
export AbstractLatentPrior, LatentModel, NonGaussianLatentPrior, hyperparameters,
    constraints, model_name, precision_logdet, IIDModel, ARModel, AR1Model, RWModel,
    RW1Model, RW2Model, BesagModel, BYM2Model, FixedEffectsModel, CombinedModel,
    component_model, SeparableModel, AutoDiffLatentPrior, StructuredLatentPrior,
    LatentFactorGroup, LocalLatentQuadratic, local_quadratic, prior_logdensity

# Formula interface
export IID, RandomWalk, AR1, AR, Besag, BYM2, Separable, Matern, build_formula_components,
    predict_cols

# Observation models and likelihoods
export ObservationModel, ObservationLikelihood, latent_dimension, conditional_distribution,
    loglik, loggrad, loghessian, pointwise_loglik, pointwise_loglik!, ZeroLikelihood,
    AutoDiffObservationModel, AutoDiffLikelihood, LinearlyTransformedObservationModel,
    LinearlyTransformedLikelihood, ParameterizedMatrix, ParameterizedOffset,
    NonlinearLeastSquaresModel, NonlinearLeastSquaresLikelihood,
    StructuredObservationModel, ObsFactorGroup, CompositeObservations,
    CompositeObservationModel, CompositeLikelihood

# Exponential families
export ExponentialFamily, LinkFunction, IdentityLink, LogLink, LogitLink,
    BinomialObservations, PoissonObservations, NegativeBinomialObservations

# SPDEs and SPDE-based models
export SPDE, ndim, intrinsic_dim, discretize, MaternSPDE, AdvectionDiffusionSPDE,
    MaternModel, BarrierModel, barrier_triangles

# FEM discretizations and meshes
export FEMDiscretization, evaluation_matrix, node_selection_matrix, derivative_matrices,
    second_derivative_matrices, generate_mesh, create_inflated_rectangle,
    PointEvaluationObsModel, PointDerivativeObsModel, PointSecondDerivativeObsModel

# Spatiotemporal GMRFs
export AbstractSpatiotemporalGMRF, N_t, time_means, time_vars, time_stds, time_rands,
    discretization_at_time, spatial_to_spatiotemporal,
    kronecker_product_spatiotemporal_model, ConstantMeshSTGMRF,
    ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF

# Spatial utilities
export contiguity_adjacency, generate_car_model

# Solvers and marginal variances
export selinv, backward_solve, logdet_cov, AbstractVarianceStrategy, RBMCStrategy,
    BlockRBMCStrategy

# Workspaces
export GMRFWorkspace, WorkspaceGMRF, CHOLMODBackend, CliqueTreesBackend, PinDenseColumns,
    ordering_permutation, update_precision!, update_precision_values!,
    AbstractLatentWorkspace, AbstractLatentWorkspacePool, make_workspace,
    make_workspace_pool, WorkspacePool, checkout, checkin, with_workspace

# Preconditioners
export AbstractPreconditioner, FullCholeskyPreconditioner, BlockJacobiPreconditioner,
    TridiagonalBlockGaussSeidelPreconditioner,
    TridiagSymmetricBlockGaussSeidelPreconditioner, temporal_block_gauss_seidel

# Sparse approximate Cholesky (KL-optimal)
export approximate_gmrf_kl

# Graphical lasso
export graphical_lasso

# Deprecated: `condition_on_observations`, exported by its `@deprecate` in
# `arithmetic/condition/linear.jl`

# Plotting (Makie extension)
export gmrf_fem_1d_plot, gmrf_fem_1d_plot!, gmrf_fem_1d_spatiotemporal_plot,
    gmrf_fem_1d_spatiotemporal_plot!

# Public but not exported: the interface for writing custom observation likelihoods, and
# observation accessors whose generic names would clash with other packages (e.g. StatsBase).
# (`public` is a keyword from Julia 1.11 on; on 1.10 these are simply not exported.)
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public ObservationIndependence, ConditionallyIndependent, ConditionallyDependent, observation_independence"))
    eval(Meta.parse("public counts, exposure, successes, trials"))
end

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
# Last, so every function and type it names is defined
include("extension_hints.jl")

end
