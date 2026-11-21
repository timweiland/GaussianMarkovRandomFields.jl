# # Building autoregressive models
#
# ## Introduction
# In the following, we will construct a *first-order auto-regressive model*
# with Gaussian errors.
# Mathematically, this is expressed by
#
# ```math
# x_0 \sim \mathcal{N}(\mu_0, \Lambda_0), \\
# x_{t+1} = \phi x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Lambda).
# ```
#
# The latter equation is equivalent to the likelihood
# ```math
# x_{t+1} \mid x_t \sim \mathcal{N}(\phi x_t, \Lambda).
# ```
#
# Under this model, the joint distribution over $x_1, \dots, x_N$ (where $N$ is
# the number of steps) is Gaussian:
# ```math
# \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_{N-1} \\ x_N \end{pmatrix}
# \sim \mathcal{N}\left(
# \begin{pmatrix} \mu_0 \\ \phi \mu_0 \\ \phi^2 \mu_0 \\ \vdots \\
# \phi^{N-1} \mu_0 \\ \phi^N \mu_0 \end{pmatrix},
# \begin{pmatrix} \Lambda_0 & -\phi & & & & & \\
# -\phi & \Lambda + \phi^2 & -\phi & & & & \\
# & & & & & & \\
# & & \ddots & \ddots & \ddots & & \\
# & & & & & & \\
# & & & & -\phi & \Lambda + \phi^2 & -\phi \\
# & & & & & -\phi & \Lambda
# \end{pmatrix}^{-1}
# \right).
# ```
# The first-order Markov structure of the model results in a tridiagonal
# precision matrix. Thus, if we work with this Gaussian distribution in
# precision form (also commonly called *information form*), we gain tremendous
# computational benefits. By contrast, the *covariance matrix* of this Gaussian
# is fully dense.
#
# More generally, this package deals with *any* sparse
# precision matrix, not just tridiagonal ones. Such Gaussians with sparse
# precision matrices are called *GMRFs* (short for
# Gaussian Markov Random Fields).
#
# In the following, we construct a GMRF for the above first-order autoregressive
# model first manually by computing the mean and precision, and then
# automatically by simply specifying the parameters of the model.

# ## Building an AR(1) model
# We begin by loading `GaussianMarkovRandomFields` and `LinearAlgebra`.
using GaussianMarkovRandomFields, LinearAlgebra

# We define a discretization of the real interval $[0, 1]$, and specify
# some example parameters for the AR(1) model:

xs = 0:0.01:1
N = length(xs)
ρ = 0.995
τ = 3.0e4

# Now we compute the mean and the precision matrix of the joint distribution.
# We explicitly declare the precision matrix as a symmetric tridiagonal matrix,
# which unlocks highly efficient linear algebra routines for the underlying
# computations.
using LinearSolve
μ = [ρ^(i - 1) for i in eachindex(xs)]
main_diag = τ * [[1.0]; repeat([1.0 + ρ^2], N - 2); [1.0]]
off_diag = τ * repeat([-ρ], N - 1)
Q = SymTridiagonal(main_diag, off_diag)
x = GMRF(μ, Q, LinearSolve.LDLtFactorization())

# A GMRF is a multivariate Gaussian, and it's compatible with
# `Distributions.jl`. We can get its mean, marginal standard deviation, and
# draw samples as follows:
using Plots, Distributions
plot(
    xs, mean(x), ribbon = 1.96 * std(x), label = "Mean ± 1.96 std",
    xlabel = "t", ylabel = "x", title = "AR(1) prior"
)
for i in 1:3
    plot!(xs, rand(x), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

# Great! Looks like an AR(1) model.
#
# But what can you do with this? Well, for example you can use it as a prior
# for Bayesian inference. If we have a likelihood of the form
# ```math
# y \mid x \sim \mathcal{N}(Ax, \Lambda_{\text{obs}}^{-1}),
# ```
# then the posterior conditioned on these observations is again a GMRF, the
# moments of which we can compute in closed form.
#
# In terms of code, the workflow for this is as follows:
# 1. Create a so-called "Observation Model". This basically defines the "category" of likelihood.
# 2. Instantiate a concrete likelihood from the model. This involves specifying concrete observations and hyperparameters.
# 3. Form a Gaussian approximation to the posterior under the prior and the observation likelihood.
#
# In the case of linear Gaussian likelihoods, the "approximation" is of course exact, which our package leverages under the hood
# by dispatching to a closed-form expression.
import Distributions
obs_model = ExponentialFamily(Distributions.Normal, indices = [26, 76]) # Model
y = [0.85, 0.71]
obs_lik = obs_model(y; σ = 0.001) # Concrete likelihood
x_cond = gaussian_approximation(x, obs_lik)

# Indeed, our model now conforms to these observations:
plot(
    xs, mean(x_cond), ribbon = 1.96 * std(x_cond), label = "Mean ± 1.96 std",
    xlabel = "t", ylabel = "x", title = "AR(1) posterior given observations"
)
for i in 1:3
    plot!(xs, rand(x_cond), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

# ## Latent models API
# Above, we constructed the precision matrix of the AR1 GMRF manually.
# But of course, GaussianMarkovRandomFields.jl also provides utilities to construct common GMRF structures automatically.
# This is implemented through so-called "Latent Models".
#
# The workflow is similar to that of Observation Models:
# 1. Construct a "Latent Model", which defines a "category" of GMRF structure.
# 2. Instantiate a concrete GMRF from the latent model. This involves specifying the concrete hyperparameter values.
latent_model = AR1Model(N)
x_ar1 = latent_model(ρ = ρ, τ = τ)

# ## Beyond first-order models: AR(2)
# You may have noticed that the AR(1) model above produces very rough samples.
# This may or may not be desirable, depending on the application.
# A first-order model only penalises how much consecutive values differ. To get
# smoother samples, we also need to penalise how much the *slope* changes, which
# requires looking two steps back. A natural choice is the second-order
# autoregression
# ```math
# x_t = 2 \rho x_{t-1} - \rho^2 x_{t-2} + \varepsilon_t,
# ```
# whose characteristic polynomial has a double root at $\rho$. As $\rho \to 1$, it
# turns into the second-order random walk $x_t - 2 x_{t-1} + x_{t-2} = \varepsilon_t$,
# a discrete analogue of integrated Brownian motion.
#
# `ARModel` parameterizes autoregressive models of any order through their partial
# autocorrelations, which keeps every parameter value stationary. For the
# recursion above, these are $2 \rho / (1 + \rho^2)$ and $-\rho^2$. We pick the
# precision `τ` such that the marginal variance roughly matches that of the AR(1)
# prior, so the two models differ only in their smoothness.
ρ₂ = 0.95
ar2_model = ARModel{2}(N)
Q_ar2 = precision_matrix(ar2_model; τ = 6.0e5, pacf1 = 2ρ₂ / (1 + ρ₂^2), pacf2 = -ρ₂^2)
x_ar2 = GMRF(μ, Q_ar2)

# The precision matrix now has two off-diagonals on each side of the diagonal.
# Let's look at one of its rows away from the boundary, normalised by its
# diagonal entry:
round.(Vector(Q_ar2[50, 48:52]) ./ Q_ar2[50, 50]; digits = 3)

# A row of the precision matrix encodes the conditional distribution of one value
# given all the others: $\mathbb{E}[x_i \mid x_{-i}] = -\sum_{j \neq i} Q_{ij} x_j / Q_{ii}$.
# Here, this is approximately
# ```math
# \mathbb{E}[x_i \mid x_{-i}] \approx \tfrac{2}{3} (x_{i-1} + x_{i+1}) - \tfrac{1}{6} (x_{i-2} + x_{i+2}).
# ```
# The negative weights on the second neighbours turn this into a local linear
# extrapolation: each value is pulled towards the line through its neighbours,
# which penalises curvature. The AR(1) conditional mean, by contrast, is a weighted
# average of the two direct neighbours, which only penalises differences.
# The effect is clearly visible in the samples:
plot(
    xs, mean(x_ar2), ribbon = 1.96 * std(x_ar2), label = "Mean ± 1.96 std",
    xlabel = "t", ylabel = "x", title = "AR(2) prior"
)
for i in 1:3
    plot!(xs, rand(x_ar2), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

# Let's see how this model fits data. We take the same observations as for the
# AR(1) model, but also add an observation for the starting point to reduce
# the uncertainty there.
obs_model = ExponentialFamily(Distributions.Normal, indices = [1, 26, 76])
y = [1.0, 0.85, 0.71]
obs_lik = obs_model(y; σ = 0.001)
x_ar2_cond = gaussian_approximation(x_ar2, obs_lik)
plot(
    xs, mean(x_ar2_cond), ribbon = 1.96 * std(x_ar2_cond), label = "Mean ± 1.96 std",
    xlabel = "t", ylabel = "x", title = "AR(2) posterior given observations"
)
for i in 1:3
    plot!(xs, rand(x_ar2_cond), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

# Compared to the AR(1) posterior, the posterior mean now bends smoothly through
# the observations instead of having kinks at them, and the samples are smooth
# curves. Between the observations, the uncertainty is larger, since smooth curves
# can swing further away from the straight line between two observations.
#
# ### A note on CARs
# Higher-order models add off-diagonals to the precision matrix, but more
# off-diagonals do not by themselves make a field smoother: what matters are their
# signs. *Conditional autoregressive* models (CARs) define the precision through a
# weighted neighbourhood graph, $Q = (D - \rho W) / \sigma$ with non-negative
# weights $W$ and $D$ the diagonal matrix of row sums of $W$ (see
# [`generate_car_model`](@ref)). Each value's conditional mean is then a weighted
# average of its neighbours, which only penalises differences between neighbours,
# no matter how many neighbours the graph has. The strength of CARs is the graph
# itself: they are a natural choice for areal data such as districts on a map. The
# tutorial [Advanced GMRF modelling for disease mapping](@ref) uses the closely
# related Besag model.

# ## Outlook
# Autoregressive models are a natural fit for regularly spaced time series. For
# spatial data, however, it is more common to model continuously through a Gaussian process. Fortunately, it
# turns out that popular Gaussian processes can be approximated quite nicely
# through GMRFs, allowing us to do the modelling in terms of a GP and the
# computations in terms of a GMRF. To learn more about this approach, check the
# tutorial on [Spatial Modelling with SPDEs](@ref).
