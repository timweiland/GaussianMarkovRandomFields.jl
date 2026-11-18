# # Automatic Differentiation for GMRF Hyperparameters
#
# ## Introduction
#
# GaussianMarkovRandomFields.jl provides comprehensive automatic differentiation (AD) support
# for gradient-based inference and optimization. This tutorial demonstrates how to compute
# gradients through GMRF operations to optimize hyperparameters like precision parameters,
# mean field values, and other model parameters.
#
# Four backends have hand-written rules in this package: ForwardDiff, Zygote,
# Enzyme and Mooncake. Which of them works depends on the operation and on the
# GMRF type -- differentiating through a sparse Cholesky factorization is not
# something a general-purpose AD system can do unaided, so each supported
# combination rests on a rule written for it. ForwardDiff is the only backend
# that covers every type and every operation, and it is what we use below.
#
# The [Automatic Differentiation Reference](@ref) has the full support matrix,
# checked against finite differences. Combinations without a rule raise an
# explicit error rather than returning a wrong number.
#
# !!! note "A different question: your own likelihoods"
#     The matrix above is about differentiating *through* the package's own
#     linear algebra. When you supply your own log-likelihood to
#     `AutoDiffObservationModel`, the package differentiates your function
#     through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
#     instead, so any DI-compatible backend can be passed there -- no rule needed.
#     Whether a given backend can differentiate your function then depends on that
#     backend and on your code rather than on this package.

# ## Basic Setup
#
# We'll start by loading the required packages:

using GaussianMarkovRandomFields
using DifferentiationInterface
using ForwardDiff, Zygote, Enzyme
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
y_obs = PoissonObservations(rand(conditional_distribution(obs_model, x_latent)))

println("Generated $n observations with τ = $τ_true, μ = $μ_true")

# ## Computing Gradients with DifferentiationInterface
#
# Now we'll define an objective function that maps hyperparameters to a scalar loss,
# and compute its gradient with ForwardDiff. For comparison, we also compute it with
# the reverse-mode backends Zygote and Enzyme.
#
# The objective function takes hyperparameters [log_τ, log_μ], builds a GMRF prior,
# computes a Gaussian approximation to the posterior, and returns the negative log
# marginal likelihood.
function objective(θ::AbstractVector, y::PoissonObservations, n::Int)
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

# Compute gradient with ForwardDiff
grad_forwarddiff = DifferentiationInterface.gradient(
    θ -> objective(θ, y_obs, n),
    AutoForwardDiff(),
    θ_init
)

println("ForwardDiff gradient computed, norm: $(norm(grad_forwarddiff))")

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
max_diff = max(
    maximum(abs.(grad_zygote - grad_forwarddiff)),
    maximum(abs.(grad_enzyme - grad_forwarddiff))
)
println("Maximum difference between backends: $(max_diff)")

# ## Optimization with Optim.jl
#
# We can use these gradients with optimization libraries like Optim.jl to find
# maximum a posteriori (MAP) estimates. Optim.jl allows you to choose the
# backend used for automatic differentation through its `autodiff` parameter.

using Optim

# Optimize using L-BFGS with ForwardDiff-based autodiff
result = optimize(
    θ -> objective(θ, y_obs, n),
    θ_init,
    LBFGS(; alphaguess = Optim.LineSearches.InitialStatic(; alpha = 0.001)),
    autodiff = AutoForwardDiff()
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
# This same advice applies here. ForwardDiff, Zygote, and Enzyme all support
# AD through Gaussian approximations. Constrained priors (e.g. RW, Besag) are
# supported by ForwardDiff and Zygote everywhere, and by Enzyme on Julia 1.12.
#
# Coverage differs between backends, though: which operations and GMRF types each
# one handles varies, and a few combinations need a recent Julia version. The
# [Automatic Differentiation Reference](@ref) has the support matrix, checked
# against finite differences — worth a look before settling on a backend for a
# long-running fit. For the model in this tutorial, all three backends agree.
#
# In practice, our recommendation is: start with ForwardDiff. It supports every
# GMRF type and operation, and with a handful of hyperparameters forward mode is
# also the right shape for the problem. Switch to a reverse-mode backend only when
# the number of inputs grows large, after checking the support matrix.
#
# Among the reverse-mode backends, Zygote has low compilation times and works in
# most cases. Enzyme incurs large compilation overheads and may not work in some
# situations, but once compiled it is generally much faster than Zygote.
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
# GaussianMarkovRandomFields.jl provides custom rules for common GMRF workflows.
# For the combinations in the support matrix, you should not have to worry about
# the details of these rules. If a combination you need is missing or gives wrong
# results, please open an issue on GitHub.
# For Enzyme, as mentioned above, pay extra attention to type stability.
#
# For more details on AD implementation and advanced usage, see the
# [Automatic Differentiation Reference](../reference/autodiff.md).
