using SparseArrays
using LinearAlgebra
using LinearSolve

export RWModel, RW1Model, RW2Model

_default_rw_alg(::Val{1}) = LDLtFactorization()
_default_rw_alg(::Val) = CHOLMODFactorization()

"""
    _difference_operator(n::Int, ::Val{Order}) where {Order}

Build the k-th order difference operator matrix D_k of size (n-k) x n as a sparse matrix.

D_1 has rows [-1, 1] (first differences), and D_k = D_1(n-k+1) * D_{k-1}(n) recursively.
"""
function _difference_operator(n::Int, ::Val{Order}) where {Order}
    # Base case: first-order difference operator (n-1) x n
    D = spzeros(n - 1, n)
    for i in 1:(n - 1)
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    end

    # Apply first-difference recursively for higher orders
    for _ in 2:Order
        m = size(D, 1)
        D1 = spzeros(m - 1, m)
        for i in 1:(m - 1)
            D1[i, i] = -1.0
            D1[i, i + 1] = 1.0
        end
        D = D1 * D
    end

    return D
end

"""
    RWModel{Order}(n::Int; regularization=1e-5, alg=<auto>, additional_constraints=nothing, scale_model=false)

A random walk latent model of arbitrary order for constructing intrinsic GMRFs.

The RW model of order k represents a process where k-th order differences are
i.i.d. Gaussian: Δᵏx[i] ~ N(0, τ⁻¹). This creates a singular precision matrix
Q = τ * Dₖ'Dₖ with rank n-k, where Dₖ is the k-th order difference operator.

# Variance scaling
With `scale_model = true`, the intrinsic precision is normalized following Sørbye & Rue
(2014): the precision is multiplied by a constant so the geometric mean of the marginal
variances (of the constrained model) is 1. This makes `τ` interpretable as a precision and
comparable across `n` and across orders, so a prior on `τ` (e.g. `PCPrior.Precision`)
transfers between models. Unscaled (`scale_model = false`, the default) reproduces the
previous behavior, where `τ = 1` implies an `n`- and order-dependent marginal variance.
This is the same normalization `BesagModel`/`BYM2Model` apply.

# Orders
- **Order 1 (RW1)**: First differences x[i+1] - x[i] ~ N(0, τ⁻¹). Tridiagonal precision.
- **Order 2 (RW2)**: Second differences x[i+2] - 2x[i+1] + x[i] ~ N(0, τ⁻¹). Pentadiagonal precision.
- **Order k**: k-th differences. Precision has bandwidth k.

Since the precision matrix is singular (rank n-k), the model is handled as an
intrinsic GMRF with k constraints from the polynomial null space (degrees 0, 1, ..., k-1).

# Type Aliases
- `RW1Model = RWModel{1}` — first-order random walk
- `RW2Model = RWModel{2}` — second-order random walk

# Hyperparameters
- `τ`: Precision parameter (τ > 0)

# Fields
- `n::Int`: Length of the process
- `regularization::Float64`: Small value added to diagonal after scaling (default 1e-5). Keep it small when `scale_model = true`: the variance normalization is computed at a negligible internal regularization, so a large `regularization` would shift the realized marginal variances away from the normalized scale.
- `alg::Alg`: LinearSolve algorithm (default: `LDLtFactorization()` for Order=1, `CHOLMODFactorization()` for Order≥2)
- `additional_constraints::C`: Optional additional constraints beyond the required null space constraints
- `scale_factor::Float64`: Sørbye–Rue variance-normalization factor multiplying the intrinsic precision (`1.0` when `scale_model = false`)

# Example
```julia
# First-order random walk
model = RW1Model(100)
gmrf = model(τ=1.0)

# Second-order random walk (smoother)
model2 = RW2Model(100)
gmrf2 = model2(τ=1.0)

# Third-order random walk
model3 = RWModel{3}(100)
gmrf3 = model3(τ=1.0)
```
"""
struct RWModel{Order, Alg, C, L} <: LatentModel
    n::Int
    regularization::Float64
    alg::Alg
    additional_constraints::C
    levels::L
    scale_factor::Float64

    function RWModel{Order, Alg, C, L}(n::Int, regularization::Float64, alg::Alg, additional_constraints::C, levels::L, scale_factor::Float64) where {Order, Alg, C, L}
        Order isa Int && Order >= 1 || throw(ArgumentError("Order must be a positive integer, got Order=$Order"))
        n > Order || throw(ArgumentError("RW$Order requires length n > $Order, got n=$n"))
        regularization >= 0 || throw(ArgumentError("Regularization must be non-negative, got $regularization"))
        scale_factor > 0 || throw(ArgumentError("scale_factor must be positive, got $scale_factor"))
        return new{Order, Alg, C, L}(n, regularization, alg, additional_constraints, levels, scale_factor)
    end
end

function RWModel{Order}(n::Int; regularization::Float64 = 1.0e-5, alg = _default_rw_alg(Val(Order)), additional_constraints = nothing, levels = nothing, scale_model::Bool = false) where {Order}
    if additional_constraints === :sumtozero
        throw(ArgumentError("RWModel already includes null space constraints by default. Use additional_constraints for extra constraints beyond the built-in ones."))
    end

    processed_additional = _process_constraint(additional_constraints, n)
    scale_factor = if scale_model
        n > Order || throw(ArgumentError("RW$Order requires length n > $Order, got n=$n"))
        _rw_scale_factor(Val(Order), n)
    else
        1.0
    end
    return RWModel{Order, typeof(alg), typeof(processed_additional), typeof(levels)}(n, regularization, alg, processed_additional, levels, scale_factor)
end

# Polynomial null-space constraint matrix of D_kᵀD_k: rows j^d (d = 0, …, Order-1).
function _rw_nullspace_constraints(n::Int, ::Val{Order}) where {Order}
    A = zeros(Order, n)
    for d in 0:(Order - 1), j in 1:n
        A[d + 1, j] = Float64(j)^d
    end
    return A
end

# Tiny fixed regularization that makes the singular `DₖᵀDₖ` factorizable for the
# scale-factor variance solve. Decoupled from the model's own `regularization`, matching
# `BesagModel`'s `_compute_normalization`; the null-space constraint removes the actual
# singular directions, so this only needs to nudge the factorization.
const _RW_SCALE_REG = 1.0e-5

# Sørbye & Rue (2014) variance scaling. Returns the factor `c` such that the precision
# `c · DₖᵀDₖ` (under the model's null-space constraints) has geometric-mean marginal
# variance 1. That factor is the geometric mean of the marginal variances of the *unscaled*
# constrained intrinsic model — equivalently the geomean of `diag(Q⁺)`, the Moore–Penrose
# pseudoinverse diagonal. Computed once at construction.
function _rw_scale_factor(::Val{Order}, n::Int) where {Order}
    D = _difference_operator(n, Val(Order))
    Q = sparse(D' * D) + _RW_SCALE_REG * I
    A = _rw_nullspace_constraints(n, Val(Order))
    x = ConstrainedGMRF(GMRF(zeros(n), Q), A, zeros(Order))
    return _geomean(var(x))
end

"""Backward-compatible alias for `RWModel{1}`."""
const RW1Model = RWModel{1}

"""Convenience alias for `RWModel{2}`."""
const RW2Model = RWModel{2}

function Base.length(model::RWModel)
    return model.n
end

function hyperparameters(model::RWModel)
    return (τ = Real,)
end

function _validate_rw_parameters(; τ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    return nothing
end

# Specialized precision matrix for Order=1: returns SymTridiagonal
function precision_matrix(model::RWModel{1}; τ::Real, kwargs...)
    _validate_rw_parameters(; τ = τ)

    n = model.n
    T = promote_type(typeof(τ), Float64)
    sτ = model.scale_factor * τ   # Sørbye–Rue variance scaling (factor is 1 when unscaled)

    main_diag = map(1:n) do i
        base_val = (i == 1 || i == n) ? T(1) : T(2)
        base_val * sτ + model.regularization
    end

    off_diag = fill(-T(sτ), n - 1)

    return SymTridiagonal(main_diag, off_diag)
end

# Generic precision matrix for Order >= 2: returns SparseMatrixCSC
function precision_matrix(model::RWModel{Order}; τ::Real, kwargs...) where {Order}
    _validate_rw_parameters(; τ = τ)

    n = model.n
    T = promote_type(typeof(τ), Float64)

    D = _difference_operator(n, Val(Order))
    Q = T(model.scale_factor * τ) * (D' * D) + model.regularization * sparse(T(1) * I, n, n)

    return Q
end

function mean(model::RWModel; kwargs...)
    return zeros(model.n)
end

function constraints(model::RWModel{Order}; kwargs...) where {Order}
    n = model.n

    # Null space of D_k'*D_k: polynomials of degree 0, 1, ..., k-1 (row d+1 is j^d).
    A_nullspace = _rw_nullspace_constraints(n, Val(Order))
    e_nullspace = zeros(Order)

    if model.additional_constraints === nothing
        return (A_nullspace, e_nullspace)
    end

    A_add, e_add = model.additional_constraints
    return (vcat(A_nullspace, A_add), vcat(e_nullspace, e_add))
end

function model_name(::RWModel{Order}) where {Order}
    return Symbol("rw$Order")
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
