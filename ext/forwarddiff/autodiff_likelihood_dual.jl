# IFT path for AutoDiffLikelihood with Dual-valued hyperparameters.
#
# When a hyperparameter stored on the likelihood carries `ForwardDiff.Dual`
# partials (typical for outer-AD callers wrapping `gaussian_approximation`
# in a `hyperparameter_logpdf(θ)` function), the standard
# `gaussian_approximation` Newton iteration would try to mutate the
# workspace's `Float64`-typed `Q.nzval` with Dual values. We sidestep by:
#
#   1. Stripping Duals from `obs_lik.hyperparams` and running primal Newton
#      to convergence on a plain Float64 likelihood.
#   2. Computing ∂(∇_x loglik)/∂θ at the converged x* via *exact AD*: lift
#      x* to a `Dual{outer_tag, Float64, N}` with zero outer-partials, call
#      `loggrad`, and read partials off the result. The closure carries
#      Dual hyperparams so the output naturally captures θ-direction
#      partials.
#   3. Solving the IFT linear system Q_post · dx*/dθ_j = grad_dθ_j with the
#      primal posterior Cholesky for each partial direction.
#   4. Assembling a Dual posterior mean from primal x_star + IFT-derived
#      partials. Computing the *total* `dH/dθ` via one `loghessian` call on
#      Dual x* (carrying dx/dθ partials) + Dual hyperparams. Writing into
#      the primal Q's exact sparse structure (preserves `colptr`/`rowval`
#      bit-for-bit so workspace pattern checks aren't tripped).
#
# This path uses no finite differences anywhere — all derivatives are
# computed via AD.

# ----------------------------------------------------------------------------
# Strip / detect helpers — extend the main-src defaults to recognise Dual
# scalars and Dual-eltype arrays.
# ----------------------------------------------------------------------------

GMRFs._strip_ad_partials(x::ForwardDiff.Dual) = ForwardDiff.value(x)
GMRFs._strip_ad_partials(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)

GMRFs._carries_ad_partials(::ForwardDiff.Dual) = true
GMRFs._carries_ad_partials(x::AbstractArray{<:ForwardDiff.Dual}) = true

# ----------------------------------------------------------------------------
# Outer-Dual introspection
# ----------------------------------------------------------------------------

# Locate the (Tag, N) carried by Dual hyperparams. Assumes a single outer
# ForwardDiff pass — all Duals share the same Tag and number of partials.
function _outer_tag_and_npartials(hp::NamedTuple)
    for v in values(hp)
        if v isa ForwardDiff.Dual
            return ForwardDiff.tagtype(typeof(v)), ForwardDiff.npartials(typeof(v))
        elseif v isa AbstractArray && eltype(v) <: ForwardDiff.Dual
            return ForwardDiff.tagtype(eltype(v)), ForwardDiff.npartials(eltype(v))
        end
    end
    return error("no Dual hyperparam in $(keys(hp))")
end

# ----------------------------------------------------------------------------
# Likelihood rebuild + primal stripping
# ----------------------------------------------------------------------------

# Build a fresh AutoDiffLikelihood with the same wrapped function, data,
# and AD backends as `lik`, but with a different `hyperparams` NamedTuple.
# Allocates its own prep cache (since the input eltype changes with `hp`).
function _rebuild_autodiff_likelihood(lik::GMRFs.AutoDiffLikelihood, hp::NamedTuple)
    return GMRFs.AutoDiffLikelihood(
        lik.loglik_func;
        n_latent = lik.prep_cache.n_latent,
        y = lik.y,
        hyperparams = hp,
        grad_backend = lik.grad_backend,
        hessian_backend = lik.hess_backend,
        pointwise_loglik_func = lik.pointwise_loglik_func,
    )
end

# Primal-stripped likelihood: AD partials removed from each hyperparam.
_primal_autodiff_likelihood(lik::GMRFs.AutoDiffLikelihood) =
    _rebuild_autodiff_likelihood(lik, GMRFs._strip_ad_partials_hyperparams(lik.hyperparams))
