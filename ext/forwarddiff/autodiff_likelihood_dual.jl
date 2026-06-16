# Helpers for AutoDiffLikelihood values whose hyperparameters carry
# ForwardDiff.Dual partials. The unified workspace IFT implementation
# (which consumes these helpers) lives in autodiff_likelihood_ift.jl.

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

# Locate the (Tag, N) carried by Dual hyperparams. The IFT path requires
# all Dual hyperparams to share a single outer-AD pass — same Tag and
# same number of partials. Mixing Duals from independent outer passes (or
# different chunk sizes) would silently misread partials in step 2/5.
function _outer_tag_and_npartials(hp::NamedTuple)
    Tag = nothing
    N = nothing
    for (k, v) in pairs(hp)
        T_v, N_v = _hp_entry_tag_npartials(v)
        T_v === nothing && continue
        if Tag === nothing
            Tag, N = T_v, N_v
        elseif T_v !== Tag || N_v != N
            # COV_EXCL_START
            throw(
                ArgumentError(
                    "AutoDiffLikelihood IFT path: hyperparams carry Duals from " *
                        "different outer-AD passes (entry `$k` has Tag=$T_v / N=$N_v, " *
                        "expected Tag=$Tag / N=$N). All Dual hyperparams must come " *
                        "from a single outer ForwardDiff pass."
                )
            )
            # COV_EXCL_STOP
        end
    end
    Tag === nothing && throw(ArgumentError("no Dual hyperparam in $(keys(hp))")) # COV_EXCL_LINE
    return Tag, N
end

_hp_entry_tag_npartials(v::ForwardDiff.Dual) =
    ForwardDiff.tagtype(typeof(v)), ForwardDiff.npartials(typeof(v))
_hp_entry_tag_npartials(v::AbstractArray{<:ForwardDiff.Dual}) =
    ForwardDiff.tagtype(eltype(v)), ForwardDiff.npartials(eltype(v))
_hp_entry_tag_npartials(::Any) = (nothing, nothing)

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
        diagonal_hessian_safe = lik.diagonal_hessian_safe,
    )
end

# Primal-stripped likelihood: AD partials removed from each hyperparam.
_primal_autodiff_likelihood(lik::GMRFs.AutoDiffLikelihood) =
    _rebuild_autodiff_likelihood(lik, GMRFs._strip_ad_partials_hyperparams(lik.hyperparams))

# Extend `_primal_obs_lik` (defined in common.jl for the per-channel Dual
# likelihoods) to recognise AutoDiffLikelihood and CompositeLikelihood. The
# unified IFT path uses this to strip the Dual-bearing lik down to a Float64
# version for the primal Newton pass.
_primal_obs_lik(lik::GMRFs.AutoDiffLikelihood) =
    _is_dual_autodifflik(lik) ? _primal_autodiff_likelihood(lik) : lik

function _primal_obs_lik(lik::GMRFs.CompositeLikelihood)
    return GMRFs.CompositeLikelihood(map(_primal_obs_lik, lik.components))
end

function _primal_obs_lik(lik::GMRFs.LinearlyTransformedLikelihood)
    return GMRFs.LinearlyTransformedLikelihood(
        _primal_obs_lik(lik.base_likelihood),
        _strip_matrix_partials(lik.design_matrix),
        _strip_offset_partials(lik.offset),
    )
end

# Strip Dual partials from a (possibly θ-dependent) design matrix.
_strip_matrix_partials(A::AbstractMatrix{<:ForwardDiff.Dual}) = ForwardDiff.value.(A)
_strip_matrix_partials(A::AbstractMatrix) = A

# Strip Dual partials from a (possibly θ-dependent) affine offset.
_strip_offset_partials(::Nothing) = nothing
_strip_offset_partials(b::AbstractVector{<:ForwardDiff.Dual}) = ForwardDiff.value.(b)
_strip_offset_partials(b::AbstractVector) = b

# Issue #157: when the latent field is Dual-valued (differentiating through
# `gaussian_approximation` w.r.t. hyperparameters), `hess_η = loghessian(η, base)` is
# `Diagonal{<:ForwardDiff.Dual}`, and the generic `Symmetric(A' * hess_η * A)` — with a
# lazy `Adjoint` `A'` — drops into an allocation-heavy generic fallback that is ~70–90×
# slower than materializing the transpose and right-associating. This Dual-specialized
# method does the latter. It is numerically identical to the generic form up to
# floating-point reassociation, and the offset is folded in identically via
# `_linear_predictor` (η = A·x + b is affine in x, so ∂η/∂x = A and the offset only
# shifts the point at which `hess_η` is evaluated). The `Float64` generic method is left
# untouched on purpose: there the lazy `Adjoint` already hits an optimized `SparseArrays`
# kernel and materializing would be marginally slower.
function GMRFs.loghessian(
        x_full::AbstractVector{<:ForwardDiff.Dual},
        ltlik::GMRFs.LinearlyTransformedLikelihood,
    )
    η = GMRFs._linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)
    hess_η = GMRFs.loghessian(η, ltlik.base_likelihood)
    A = ltlik.design_matrix
    return Symmetric(sparse(transpose(A)) * (hess_η * A))
end

# NonlinearLeastSquaresLikelihood: strip the σ-derived numeric fields (Dual when σ
# carried partials) and the stored residual hyperparameters (Dual when an `f(x; θ...)`
# hyperparameter carried partials). `f` and the cached backends (`jac_backend`,
# `hess_backend`) are unchanged; they already hold primal-detected sparsity patterns, so
# they are valid for the primal pass.
function _primal_obs_lik(lik::GMRFs.NonlinearLeastSquaresLikelihood)
    (GMRFs._hp_carries_ad_partials(lik.hyperparams) || GMRFs._carries_ad_partials(lik.inv_σ²)) || return lik
    return GMRFs.NonlinearLeastSquaresLikelihood(
        lik.f,
        ForwardDiff.value.(lik.y),
        ForwardDiff.value.(lik.inv_σ²),
        ForwardDiff.value(lik.log_const),
        lik.jac_backend,
        lik.hess_backend,
        GMRFs._strip_ad_partials_hyperparams(lik.hyperparams),
    )
end
