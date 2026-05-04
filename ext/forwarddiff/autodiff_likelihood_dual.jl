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
#   2. For each outer-FD partial direction, computing
#      ∂(∇_x loglik)/∂θ at the converged x* via central finite differences
#      on the stored hyperparams (no nested AD — outer θ-direction is FD,
#      inner ∇_x is primal).
#   3. Solving the IFT linear system Q_post · dx*/dθ_j = grad_dθ_j with the
#      primal posterior Cholesky for each partial direction.
#   4. Assembling a Dual posterior mean from primal x_star + per-direction
#      partials, and a Dual posterior precision by writing into the primal
#      Q's exact sparse structure (preserves `colptr`/`rowval` bit-for-bit
#      so workspace pattern checks aren't tripped).

# ----------------------------------------------------------------------------
# Strip / detect helpers — extend the main-src defaults to recognise Dual
# scalars and Dual-eltype arrays.
# ----------------------------------------------------------------------------

GMRFs._strip_dual_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
GMRFs._strip_dual_value(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)

GMRFs._carries_dual_value(::ForwardDiff.Dual) = true
GMRFs._carries_dual_value(x::AbstractArray{<:ForwardDiff.Dual}) = true

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

# Build a primal-stripped `hyperparams` with the j-th partial direction added
# to each Dual entry, scaled by ε. Non-Dual entries pass through unchanged.
function _perturb_along_partial(hp_dual::NamedTuple, j::Int, ε::Float64)
    pairs = map(keys(hp_dual)) do name
        v = getfield(hp_dual, name)
        name => _perturb_one(v, j, ε)
    end
    return (; pairs...)
end
_perturb_one(v::ForwardDiff.Dual, j, ε) = ForwardDiff.value(v) + ε * ForwardDiff.partials(v, j)
function _perturb_one(v::AbstractArray{<:ForwardDiff.Dual}, j, ε)
    # `map` preserves shape for AbstractArrays; a comprehension would flatten
    # to a Vector and break broadcasting against the same-shape values for
    # Matrix or higher-rank hyperparameters.
    return ForwardDiff.value.(v) .+ ε .* map(x -> ForwardDiff.partials(x, j), v)
end
_perturb_one(v, j, ε) = v

# Build a primal-eltype version of the AutoDiffLikelihood with stripped
# hyperparameters. The new instance has its own prep cache (Float64-only),
# but reuses everything else from the original.
function _primal_autodiff_likelihood(lik::GMRFs.AutoDiffLikelihood)
    primal_hp = GMRFs._strip_dual_hyperparams(lik.hyperparams)
    return GMRFs.AutoDiffLikelihood(
        lik.loglik_func;
        n_latent = lik.prep_cache.n_latent,
        y = lik.y,
        hyperparams = primal_hp,
        grad_backend = lik.grad_backend,
        hessian_backend = lik.hess_backend,
        pointwise_loglik_func = lik.pointwise_loglik_func,
    )
end

# Build a perturbed AutoDiffLikelihood for a given partial direction.
function _perturbed_autodiff_likelihood(lik::GMRFs.AutoDiffLikelihood, j::Int, ε::Float64)
    perturbed_hp = _perturb_along_partial(lik.hyperparams, j, ε)
    return GMRFs.AutoDiffLikelihood(
        lik.loglik_func;
        n_latent = lik.prep_cache.n_latent,
        y = lik.y,
        hyperparams = perturbed_hp,
        grad_backend = lik.grad_backend,
        hessian_backend = lik.hess_backend,
        pointwise_loglik_func = lik.pointwise_loglik_func,
    )
end
