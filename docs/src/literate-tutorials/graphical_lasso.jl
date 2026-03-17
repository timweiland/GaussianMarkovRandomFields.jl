# # Learning GMRFs from data with the graphical lasso
#
# ## Introduction
#
# In many applications, we observe data that we believe has been generated
# by a process with sparse conditional independence structure, but we
# do not know the exact precision matrix. The
# [graphical lasso](https://en.wikipedia.org/wiki/Graphical_lasso) lets us
# *learn* a GMRF directly from sample data by estimating a sparse precision
# matrix.
#
# The graphical lasso solves the following optimization problem:
# ```math
# \max_{\Omega \succ 0} \; \log\det \Omega - \operatorname{tr}(S\Omega) - \lambda \lVert \Omega \rVert_1
# ```
# where ``S`` is the sample covariance matrix and ``\lambda`` controls the
# sparsity of the estimated precision matrix ``\Omega``.
#
# The implementation follows [Zhang2018](@cite), combining soft-thresholding
# of the sample covariance with maximum-determinant positive-definite matrix
# completion via chordal graph techniques.
#
# ## Synthetic example
#
# Let's generate data from a known sparse precision matrix and see if we
# can recover it.

using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Random

rng = Random.MersenneTwister(42)

# We construct a diagonally dominant sparse precision matrix.

n = 200
A = sprand(rng, n, n, 0.02)
A = A + A'

for i in 1:n
    A[i, i] = sum(abs, A[:, i]) + 1.0
end

M = Hermitian(A, :L)
A[1:50, 1:50]

# Now we draw samples from the corresponding Gaussian distribution.

m = 2000
X = copy(transpose(cholesky(M).L \ randn(rng, n, m)))

# ## Estimating the precision matrix
#
# We apply [`graphical_lasso`](@ref) with a scalar threshold ``\lambda``.
# The threshold controls the sparsity of the result: larger values yield
# sparser precision matrices.

lambda = 0.03
gmrf = graphical_lasso(X, lambda)

# The result is a [`GMRF`](@ref) whose precision matrix is a sparse
# estimate of the true precision.

P_est = precision_matrix(gmrf)
P_est[1:50, 1:50]

# ## Restricted graphical lasso
#
# If we have prior knowledge about the sparsity pattern (e.g., from
# a graph structure), we can pass a sparse matrix as the threshold.
# The algorithm then only searches for precision matrices within that
# sparsity pattern, with per-entry penalty values.

Lambda = copy(A)
Lambda.nzval .= lambda

gmrf_restricted = graphical_lasso(X, Lambda)
P_restricted = precision_matrix(gmrf_restricted)
P_restricted[1:50, 1:50]
