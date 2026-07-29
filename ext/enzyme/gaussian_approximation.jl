# COV_EXCL_START
# `gaussian_approximation` finds the mode x* of the posterior by Fisher scoring.
# Differentiating the solver loop would be slow and unstable, so this rule applies
# the Implicit Function Theorem to the optimality condition instead, mirroring the
# backend-agnostic rrule in `src/autodiff/gaussian_approximation.jl` step for step.
#
# With  g(x, p) = ∇ₓ neg_log_posterior = Q(x - μ) - loggrad(x, obs_lik)  and
# g(x*, p) = 0, the posterior is  (μ_post, Q_post) = (x*, Q - loghessian(x*, ·)).
# Given cotangents (μ̄, Q̄) on that posterior:
#
#   1. Q̄ reaches x* through loghessian:  x̄ += ∂/∂x* ⟨-Q̄, loghessian(x*, ·)⟩
#   2. μ̄ reaches x* directly, since μ_post = x*
#   3. IFT:  λ = Q_post⁻¹ (μ̄ + x̄),  then every parameter picks up -λᵀ ∂g/∂p
#   4. Q̄ also reaches Q_prior directly, since Q_post = Q_prior - loghessian
#
# The previous version of this rule got two things wrong. It read the posterior's
# precision cotangent as a plain gradient matrix when the posterior stores
# `Symmetric(Q, :U)` and the prior a bare `SparseMatrixCSC` — conventions that
# disagree by a factor of two on off-diagonals — and it accumulated into sparse
# shadows with `.+=`, which reshapes their buffers. Both are handled here by
# `cotangent_matrix` and `accumulate_on_pattern!` respectively.
#
# Step 3's VJP against the *prior* is written out in closed form rather than
# delegated to a nested `Enzyme.autodiff`. Differentiating `∇ₓ_neg_log_posterior`
# w.r.t. a whole GMRF struct would drag Enzyme through the LinearSolve cache and
# its CHOLMOD pointers; the derivatives are two lines of algebra, so there is no
# reason to. Only the observation likelihood — whose hyperparameters are what a
# marginal-likelihood optimisation is actually after — needs a nested call.

# Priors whose IFT solve and cotangent layout are covered here.
const GA_SUPPORTED = Union{GMRF, ConstrainedGMRF, ChordalGMRF, WorkspaceGMRF}

"""
    ift_solve(posterior, x̄, prior) -> λ

Solve `Q_post λ = x̄`, the linear system at the heart of the IFT step.

`GMRF`, `ConstrainedGMRF` and `ChordalGMRF` reuse the main package's
`_ift_solve`, which also projects the step onto the constraint null space when
there is one. `WorkspaceGMRF` keeps its factorization in the workspace instead of
a `LinearSolve` cache, so it goes through `workspace_solve` — matching what the
ChainRules workspace rule does.
"""
ift_solve(posterior, x̄, prior) = GMRFs._ift_solve(posterior, x̄, prior)

function ift_solve(posterior::WorkspaceGMRF, x̄, prior)
    GMRFs.ensure_loaded!(posterior)
    return GMRFs.workspace_solve(posterior.workspace, x̄)
end

"""
    ga_base(d)

The GMRF whose `μ` and `Q` the approximation is actually parameterised by —
`ConstrainedGMRF` keeps them one level down. Extends the main package's
`_base_gmrf`, which has no `WorkspaceGMRF` method because the workspace rrule
never needed one.
"""
ga_base(d) = _base_gmrf(d)
ga_base(d::WorkspaceGMRF) = d

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        prior_gmrf::Annotation{<:GA_SUPPORTED},
        obs_lik::Annotation
    ) where {RT}
    # Gauss-Newton likelihoods carry a sparse Jacobian that reverse mode cannot
    # differentiate; refuse with the package's shared message rather than failing
    # somewhere inside Enzyme.
    GMRFs._has_gauss_newton_jacobian(obs_lik.val) &&
        GMRFs._reverse_mode_gauss_newton_error()
    enzyme_check_supported("gaussian_approximation", prior_gmrf.val)

    posterior = gaussian_approximation(prior_gmrf.val, obs_lik.val)

    if EnzymeRules.needs_shadow(config)
        shadow = zero_gmrf_shadow(posterior)
    else
        shadow = nothing
    end

    tape = (x_star = mean(ga_base(posterior)), posterior = posterior, shadow = shadow)

    return EnzymeRules.AugmentedReturn(posterior, return_shadow(RT, shadow), tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(gaussian_approximation)},
        ::Type{RT},
        tape,
        prior_gmrf::Annotation{<:GA_SUPPORTED},
        obs_lik::Annotation
    ) where {RT}
    shadow = tape.shadow
    shadow === nothing && return (nothing, nothing)

    posterior = tape.posterior
    x_star = tape.x_star

    μ̄, Q̄ = posterior_cotangents(shadow, posterior)
    has_Q = !all(iszero, _nonzeros(Q̄))

    # (1) + (2): assemble the cotangent on x*.
    x̄ = collect(μ̄)
    if has_Q
        x̄_from_hess = zero(x_star)
        Enzyme.autodiff(
            Reverse, Const(hessian_contraction), Active,
            Duplicated(collect(x_star), x̄_from_hess),
            obs_lik,
            Const(Q̄),
        )
        x̄ .+= x̄_from_hess
    end

    # (3): λ = Q_post⁻¹ x̄, projected onto the constraint null space when the
    # prior carries constraints — the same solve the Newton loop itself uses.
    λ = ift_solve(posterior, x̄, prior_gmrf.val)

    base_prior = ga_base(prior_gmrf.val)

    if is_active(prior_gmrf)
        base_shadow = shadow_of(prior_gmrf)
        Q = precision_matrix(base_prior)
        r = collect(x_star) .- collect(mean(base_prior))

        # ⟨-λ, ∂g/∂μ⟩ with g = Q(x - μ) - loggrad  ⇒  ∂/∂μ = Qλ.
        shadow_mean(base_shadow) .+= Q * λ

        # ∂/∂Q_ij of ⟨-λ, Q(x-μ)⟩ is -λᵢ rⱼ. Symmetrised because the stored
        # entries of a triangle-backed precision each move two positions of the
        # effective matrix; for full storage the symmetrisation is a no-op on the
        # final θ-gradient. (4) adds Q̄ on top, since Q_post = Q_prior - H.
        accumulate_on_pattern!(
            (i, j) -> -0.5 * (λ[i] * r[j] + λ[j] * r[i]) + (has_Q ? Q̄[i, j] : 0.0),
            shadow_precision(base_shadow),
            precision_storage(base_prior),
        )
    end

    # (3) again, for the likelihood's own hyperparameters:
    # ⟨-λ, ∂g/∂θ⟩ = ⟨-λ, -∂loggrad/∂θ⟩ = ∂⟨λ, loggrad⟩/∂θ.
    if is_active(obs_lik)
        Enzyme.autodiff(
            Reverse, Const(loggrad_contraction), Active,
            obs_lik,
            Const(collect(x_star)),
            Const(λ),
        )
    end

    zero_gmrf_shadow!(shadow)

    return (nothing, nothing)
end

# --- pieces the rule leans on ------------------------------------------------

"""
    posterior_cotangents(shadow, posterior) -> (μ̄, Q̄)

Cotangents on the approximation's mode and precision, in mathematical gradient
form (see [`cotangent_matrix`](@ref)).
"""
function posterior_cotangents(shadow, posterior::AbstractGMRF)
    μ̄ = copy(shadow_mean(shadow))
    Q̄ = cotangent_matrix(
        copy_cotangent(shadow_precision(shadow)), precision_storage(posterior)
    )
    return μ̄ .+ constrained_mean_cotangent(shadow, posterior), Q̄
end

"""
    constrained_mean_cotangent(shadow, posterior)

The part of the mode's cotangent that arrived via `mean(posterior)`.

For an unconstrained GMRF that is already on the base mean and this contributes
nothing. For a constrained one, `mean` reports `P x*` with `P = I - ÃᵀS⁻¹A`, so
the cotangent lands on `constrained_mean` and has to be pulled back with `Pᵀ =
I - AᵀS⁻¹Ã` — which differs from `P` whenever `Q ≠ I`.
"""
constrained_mean_cotangent(shadow, ::AbstractGMRF) = false

function constrained_mean_cotangent(shadow, posterior::ConstrainedGMRF)
    μ̄_c = shadow.constrained_mean
    all(iszero, μ̄_c) && return false
    v = posterior.L_c \ (posterior.A_tilde_T' * μ̄_c)
    return μ̄_c .- posterior.constraint_matrix' * v
end

# Contractions handed to the nested `Enzyme.autodiff` calls. Both take the
# constant part last so it can be marked `Const`, and both are plain top-level
# functions rather than closures — Enzyme handles those far more reliably.

"""
    hessian_contraction(x, obs_lik, Q̄) -> Real

`⟨-Q̄, loghessian(x, obs_lik)⟩`, the scalar whose gradient carries `Q̄`'s
contribution to `x*` and to the likelihood's hyperparameters.
"""
hessian_contraction(x, obs_lik, Q̄) = -contract(Q̄, loghessian(x, obs_lik))

"""
    loggrad_contraction(obs_lik, x, λ) -> Real

`⟨λ, loggrad(x, obs_lik)⟩`, the likelihood half of the IFT parameter VJP.
"""
loggrad_contraction(obs_lik, x, λ) = dot(λ, loggrad(x, obs_lik))

# `loghessian` is a `Diagonal` for every exponential family and sparse for the
# structured and autodiff likelihoods, so contract against whichever is stored
# rather than materialising a product.
#
# All three are explicit loops over a scalar accumulator, deliberately. The
# obvious `sum(G[i, j] * H[i, j] for ...)` routes through `mapfoldl`, whose
# accumulator Enzyme cannot classify — it reports "constant memory is stored to a
# differentiable variable" and refuses the whole pipeline with
# `EnzymeRuntimeActivityError`.
function contract(G, H::Diagonal)
    d = H.diag
    total = zero(eltype(d))
    @inbounds for i in eachindex(d)
        total += G[i, i] * d[i]
    end
    return total
end

function contract(G, H::AbstractSparseMatrix)
    rows = rowvals(H)
    vals = nonzeros(H)
    total = zero(eltype(vals))
    @inbounds for j in axes(H, 2), p in nzrange(H, j)
        total += G[rows[p], j] * vals[p]
    end
    return total
end

function contract(G, H::AbstractMatrix)
    total = zero(eltype(H))
    @inbounds for j in axes(H, 2), i in axes(H, 1)
        total += G[i, j] * H[i, j]
    end
    return total
end

# --- shadow lifecycle --------------------------------------------------------

"""
    zero_gmrf_shadow(d) -> typeof(d)

A cotangent buffer shaped like `d`: same fields, same sparsity patterns, all
differentiable entries zero.

The non-differentiable machinery — the `LinearSolve` cache, the factorization —
is `deepcopy`ed rather than shared. Sharing is tempting (a shadow is never
factorized) but leaves `shadow.linsolve_cache === primal.linsolve_cache`, which
Enzyme cannot prove is inactive, and the whole pipeline then fails with
`EnzymeRuntimeActivityError`.
"""
function zero_gmrf_shadow(d::GMRF)
    s = deepcopy(d)
    fill!(s.mean, 0)
    _zero_precision!(s.precision)
    return s
end

function zero_gmrf_shadow(d::WorkspaceGMRF)
    s = deepcopy(d)
    fill!(s.mean, 0)
    _zero_precision!(s.precision)
    return s
end

function zero_gmrf_shadow(d::ChordalGMRF)
    s = deepcopy(d)
    fill!(s.μ, 0)
    _zero_precision!(s.Q)
    return s
end

# Copy rather than reconstruct. Calling `ConstrainedGMRF(zeroed_base, A, e)` would
# re-derive `Ã = Q⁻¹Aᵀ` by solving against a precision that has just been zeroed,
# which is either a `PosDefException` or a silently stale factorization depending
# on whether the cache considers itself fresh. `deepcopy` keeps the primal's
# `A_tilde_T`/`L_c` — which is what the reverse pass needs to read anyway — and
# only the mutable cotangent slots are cleared.
function zero_gmrf_shadow(d::ConstrainedGMRF)
    s = deepcopy(d)
    zero_gmrf_shadow!(s)
    return s
end

function zero_gmrf_shadow!(d::AbstractGMRF)
    fill!(shadow_mean(d), 0)
    _zero_precision!(shadow_precision(d))
    return d
end

function zero_gmrf_shadow!(d::ConstrainedGMRF)
    zero_gmrf_shadow!(d.base_gmrf)
    fill!(d.constrained_mean, 0)
    return d
end

_nonzeros(A::AbstractSparseMatrix) = nonzeros(A)
_nonzeros(A::Union{Symmetric, Hermitian}) = _nonzeros(A.data)
_nonzeros(A) = A

copy_cotangent(A::Union{Symmetric, Hermitian}) =
    typeof(A).name.wrapper(copy(A.data), Symbol(A.uplo))
copy_cotangent(A) = copy(A)
# COV_EXCL_STOP
