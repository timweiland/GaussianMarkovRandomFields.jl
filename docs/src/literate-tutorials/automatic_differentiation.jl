# # Automatic Differentiation for GMRF Hyperparameters
#
# ## Introduction
#
# GaussianMarkovRandomFields.jl provides comprehensive automatic differentiation (AD) support
# for gradient-based inference and optimization. This tutorial demonstrates how to compute
# gradients through GMRF operations to optimize hyperparameters like precision parameters,
# mean field values, and other model parameters.
#
# We currently support AD via Zygote, Enzyme, and ForwardDiff.
#
# !!! note "AD may break"
#     Our current AD rules cover the most common workflows with GMRFs.
#     Less common operations may or may not work.
#     If one of these backends breaks for your use case, please open an issue.

# ## Basic Setup
#
# We'll start by loading the required packages:

using GaussianMarkovRandomFields
using DifferentiationInterface
using Zygote, Enzyme
using LinearAlgebra
using LinearSolve
using Distributions
using Random

Random.seed!(123)

# ## Example Problem: Hyperparameter Optimization
#
# Consider a problem where we have Poisson count observations and want to infer
# the hyperparameters of a simple IID (independent and identically distributed) prior.
#
# The model has two hyperparameters:
# 1. The mean parameter μ
# 2. The precision parameter τ

# Start by generating some synthetic data:
n = 50  # Number of observations
τ_true = 4.0
μ_true = 5.0

# Next, let's define the prior:
function build_prior(log_τ, log_μ, n)
    τ = exp(log_τ)
    μ = exp(log_μ)
    model = IIDModel(n)
    Q = precision_matrix(model; τ = τ)
    return GMRF(μ * ones(n), Q, LinearSolve.DiagonalFactorization())
end

# Finally, sample a ground-truth latent field and generate observations
true_gmrf = build_prior(log(τ_true), log(μ_true), n)
x_latent = rand(true_gmrf)
obs_model = ExponentialFamily(Poisson)
y_obs = rand(conditional_distribution(obs_model, x_latent))

println("Generated $n observations with τ = $τ_true, μ = $μ_true")

# ## Computing Gradients with DifferentiationInterface
#
# Now we'll define an objective function that maps hyperparameters to a scalar loss,
# and compute its gradient using both Zygote and Enzyme.
#
# The objective function takes hyperparameters [log_τ, log_μ], builds a GMRF prior,
# computes a Gaussian approximation to the posterior, and returns the negative log
# marginal likelihood.
function objective(θ::Vector{Float64}, y::Vector{Int}, n::Int)
    log_τ, log_μ = θ
    prior = build_prior(log_τ, log_μ, n)
    obs_model = ExponentialFamily(Poisson)
    likelihood = obs_model(y)
    posterior = gaussian_approximation(prior, likelihood)
    x_map = mean(posterior)
    return -logpdf(prior, x_map) - loglik(x_map, likelihood) + logpdf(posterior, x_map)
end

# Initialize hyperparameters (perturbed from truth)
θ_init = [log(τ_true) + 0.2, log(μ_true) - 0.3]

# Compute gradient with Zygote
backend_zygote = AutoZygote()
grad_zygote = DifferentiationInterface.gradient(
    θ -> objective(θ, y_obs, n),
    backend_zygote,
    θ_init
)

println("Zygote gradient computed, norm: $(norm(grad_zygote))")

# Compute gradient with Enzyme
backend_enzyme = AutoEnzyme(; function_annotation = Enzyme.Const)
grad_enzyme = DifferentiationInterface.gradient(
    θ -> objective(θ, y_obs, n),
    backend_enzyme,
    θ_init
)

println("Enzyme gradient computed, norm: $(norm(grad_enzyme))")

# Verify gradients match
max_diff = maximum(abs.(grad_zygote - grad_enzyme))
println("Maximum difference between backends: $(max_diff)")

# ## Optimization with Optim.jl
#
# We can use these gradients with optimization libraries like Optim.jl to find
# maximum a posteriori (MAP) estimates. Optim.jl allows you to choose the
# backend used for automatic differentation through its `autodiff` parameter.

using Optim

# Optimize using L-BFGS with Zygote-based autodiff
result = optimize(
    θ -> objective(θ, y_obs, n),
    θ_init,
    LBFGS(; alphaguess = Optim.LineSearches.InitialStatic(; alpha = 0.001)),
    autodiff = AutoZygote()
)

# Extract optimal parameters
θ_opt = Optim.minimizer(result)
τ_opt = exp(θ_opt[1])
μ_opt = exp(θ_opt[2])

println("\nOptimization results:")
println("  Iterations: $(result.iterations)")
println("  Estimated τ: $(round(τ_opt, digits = 2)) (true: $τ_true)")
println("  Estimated μ: $(round(μ_opt, digits = 2)) (true: $μ_true)")
println("  Converged: $(Optim.converged(result))")

# ## Choosing a Backend
#
# First, you need to choose between forward- and reverse-mode AD.
# Generally, the recommendation for AD through a function with n inputs and m outputs is:
# If n is sufficiently small or n << m, use forward-mode.
# Else, use reverse-mode.
#
# This same advice applies here, with the added caveat that ForwardDiff currently does not
# support Gaussian approximations.
# If you need to autodiff through Gaussian approximations, use Zygote or Enzyme.
#
# Both Zygote and Enzyme produce identical gradients, so the choice between them
# comes down to performance and ease of use.
#
# Zygote has low pre-compilation times and works in most cases.
# By contrast, Enzyme incurs large pre-compilation overheads and may not work in
# some situations.
# The upside is that once pre-compilation is complete, Enzyme is generally much
# faster than Zygote.
#
# In practice, our recommendation is:
# Start with Zygote for prototyping. For large-scale problems, switch to Enzyme.
#
# ## Solver Considerations
#
# Enzyme is particularly finicky when it comes to type stability.
# This causes issues when it comes to a GMRF's linear solver.
#
# Using the two-argument GMRF constructor gives you the default linear solver:
using SparseArrays
Q_sparse = sprand(10, 10, 0.3)
Q_sparse = Q_sparse + Q_sparse' + 10I  # Make symmetric positive definite
x_default_solver = GMRF(zeros(10), Q_sparse)
x_default_solver.linsolve_cache.alg

# Unfortunately, the default linear solver is not type-stable.
# It checks the type of the precision matrix at runtime and only then decides
# on an algorithm.
# In our experience, Enzyme very much does not like this behaviour.
#
# To avoid these issues, always pass a specialized linear solver to your GMRF,
# e.g. CHOLMOD for general sparse matrices:
x_cholmod = GMRF(zeros(10), Q_sparse, LinearSolve.CHOLMODFactorization())


# ## Conclusion
#
# GaussianMarkovRandomFields.jl provides custom chain rules for common GMRF
# workflows.
# As a user, you should not have to worry about the details of this.
# AD should "just work".
# If it doesn't, please open an issue on GitHub.
# For Enzyme, as mentioned above, pay extra attention to type stability.
#
# For more details on AD implementation and advanced usage, see the
# [Automatic Differentiation Reference](../reference/autodiff.md).
