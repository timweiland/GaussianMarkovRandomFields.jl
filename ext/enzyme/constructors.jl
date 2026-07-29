# `GMRF(μ, Q)` and `GMRF(μ, Q, alg)` just store their arguments, so these rules
# are only plumbing: hand Enzyme a shadow GMRF built from the argument shadows,
# then drain that shadow back into the arguments on the way out. The reason they
# exist at all is that the constructor also builds a `LinearSolve` cache, which
# Enzyme has no business walking into.

"""
    zero_shadow(A)

A zeroed cotangent buffer with the *same sparsity pattern* as `A`.

`zero(::SparseMatrixCSC)` returns a matrix with no stored entries at all, which
cannot receive an accumulation — every precision gradient written into it would
be silently dropped.
"""
zero_shadow(A::SparseMatrixCSC) = SparseMatrixCSC(
    size(A, 1), size(A, 2), copy(getcolptr(A)), copy(rowvals(A)), zero(nonzeros(A))
)
zero_shadow(A::Symmetric) = Symmetric(zero_shadow(A.data), Symbol(A.uplo))
zero_shadow(A::Hermitian) = Hermitian(zero_shadow(A.data), Symbol(A.uplo))
zero_shadow(A) = zero(A)

"""
    add_shadow!(dest, src)

Accumulate one cotangent buffer into another.

For a sparse destination this cannot be `dest .+= src`: sparse `broadcast!`
rebuilds `dest`'s `colptr`/`rowval`/`nzval`, and an Enzyme shadow has to stay
positionally aligned with its primal. When the patterns already agree — the
normal case, since both descend from the same primal — the `nzval` vectors add
directly.

Dispatch is on the *destination* only, with the source merely unwrapped. Keying
on both would leave combinations like `(SparseMatrixCSC, Symmetric)` falling
through to the generic broadcast, which is the exact corruption these methods
exist to prevent.
"""
function add_shadow!(dest::SparseMatrixCSC, src)
    s = unwrap_triangle(src)
    if s isa SparseMatrixCSC && getcolptr(dest) == getcolptr(s) && rowvals(dest) == rowvals(s)
        nonzeros(dest) .+= nonzeros(s)
    else
        accumulate_on_pattern!((i, j) -> s[i, j], dest, Val(:full))
    end
    return dest
end

add_shadow!(dest::Union{Symmetric, Hermitian}, src) =
    (add_shadow!(dest.data, unwrap_triangle(src)); dest)
add_shadow!(dest::AbstractVector, src) = (dest .+= src; dest)
add_shadow!(dest::AbstractMatrix, src) = (dest .+= unwrap_triangle(src); dest)

# Cotangent buffers wrapped in `Symmetric`/`Hermitian` are *not* symmetric
# matrices — they hold one cotangent per stored entry — so they must be read
# positionally through `.data`, never through the wrapper's mirroring `getindex`.
unwrap_triangle(A::Union{Symmetric, Hermitian}) = A.data
unwrap_triangle(A) = A

# --- GMRF(μ, Q) -------------------------------------------------------------

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        μ::Annotation{MT},
        Q::Annotation{QT}
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    primal = func.val(μ.val, Q.val)

    if EnzymeRules.needs_shadow(config)
        dres = func.val(zero_shadow(μ.val), zero_shadow(Q.val))
    else
        dres = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, dres, dres)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        tape,
        μ::Annotation{MT},
        Q::Annotation{QT}
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    _drain_gmrf_shadow!(tape, μ, Q)
    return (nothing, nothing)
end

# --- GMRF(μ, Q, alg) --------------------------------------------------------

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        μ::Annotation{MT},
        Q::Annotation{QT},
        alg::Const
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    primal = func.val(μ.val, Q.val, alg.val)

    if EnzymeRules.needs_shadow(config)
        dres = func.val(zero_shadow(μ.val), zero_shadow(Q.val), alg.val)
    else
        dres = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, dres, dres)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{GMRF}},
        ::Type{RT},
        tape,
        μ::Annotation{MT},
        Q::Annotation{QT},
        alg::Const
    ) where {RT, MT <: AbstractVector, QT <: AbstractMatrix}
    _drain_gmrf_shadow!(tape, μ, Q)
    return (nothing, nothing, nothing)
end

# Move whatever the downstream rules accumulated into the shadow GMRF back onto
# the constructor's own arguments, then reset it — Enzyme may reuse the same
# shadow across iterations of an enclosing loop.
function _drain_gmrf_shadow!(tape, μ::Annotation, Q::Annotation)
    tape === nothing && return nothing

    μ_shadow = shadow_mean(tape)
    Q_shadow = shadow_precision(tape)

    if is_active(μ)
        add_shadow!(shadow_of(μ), μ_shadow)
    end
    if is_active(Q)
        add_shadow!(shadow_of(Q), Q_shadow)
    end

    fill!(μ_shadow, zero(eltype(μ_shadow)))
    _zero_precision!(Q_shadow)
    return nothing
end

# --- WorkspaceGMRF(μ, Q) ----------------------------------------------------
#
# Without a rule here Enzyme refuses the whole pipeline with
# `EnzymeRuntimeActivityError`, because the constructor mutates state shared with
# the `GMRFWorkspace` (`_next_version!` bumps a counter, `ensure_loaded!` swaps
# the workspace's `nzval` and invalidates its factorization) and Enzyme cannot
# prove that traffic is inactive.
#
# The shadow is built through the inner constructor rather than by calling
# `WorkspaceGMRF` again: the public constructor would allocate a second workspace
# and factorize it, and a zeroed precision is not positive definite. Sharing the
# primal's workspace is safe because a shadow is only ever a cotangent buffer —
# nothing factorizes it.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{WorkspaceGMRF}},
        ::Type{RT},
        μ::Annotation{<:AbstractVector},
        Q::Annotation{<:SparseMatrixCSC}
    ) where {RT}
    primal = func.val(μ.val, Q.val)

    if EnzymeRules.needs_shadow(config)
        dres = typeof(primal)(
            zero(primal.mean), zero_shadow(primal.precision),
            primal.workspace, primal.constraints, primal.version
        )
    else
        dres = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, dres, dres)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{WorkspaceGMRF}},
        ::Type{RT},
        tape,
        μ::Annotation{<:AbstractVector},
        Q::Annotation{<:SparseMatrixCSC}
    ) where {RT}
    _drain_gmrf_shadow!(tape, μ, Q)
    return (nothing, nothing)
end

# --- ConstrainedGMRF(base, A, e) --------------------------------------------
#
# The constructor precomputes Ã = Q⁻¹Aᵀ by solving against the base GMRF's
# factorization, so without a rule Enzyme walks straight into CHOLMOD and every
# gradient through a constrained prior comes back zero.
#
# `A` and `e` define model structure rather than hyperparameters, so they receive
# no cotangent — matching the `NoTangent()` the ChainRules rrule returns for them.
# Their annotations are left open rather than pinned to `Const`: a caller that
# builds the constraint matrix inside the differentiated function hands them over
# as active, and a rule that only matched `Const` would silently not apply.
# Cotangents on the derived `log_constraint_correction` are folded into the base
# slots where they arise (see `accumulate_constraint_correction!`), so the only
# work here is moving the base cotangents — plus anything that arrived via
# `constrained_mean` — onto the constructor's own argument.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{ConstrainedGMRF}},
        ::Type{RT},
        base_gmrf::Annotation{<:AbstractGMRF},
        A::Annotation,
        e::Annotation
    ) where {RT}
    primal = func.val(base_gmrf.val, A.val, e.val)

    dres = EnzymeRules.needs_shadow(config) ? zero_gmrf_shadow(primal) : nothing

    return EnzymeRules.AugmentedReturn(primal, return_shadow(RT, dres), (primal, dres))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{Type{ConstrainedGMRF}},
        ::Type{RT},
        tape,
        base_gmrf::Annotation{<:AbstractGMRF},
        A::Annotation,
        e::Annotation
    ) where {RT}
    primal, shadow = tape
    shadow === nothing && return (nothing, nothing, nothing)

    if is_active(base_gmrf)
        bs = shadow_of(base_gmrf)
        μ̄ = shadow_mean(shadow) .+ constrained_mean_cotangent(shadow, primal)
        add_shadow!(shadow_mean(bs), μ̄)
        add_shadow!(shadow_precision(bs), shadow_precision(shadow))
    end

    zero_gmrf_shadow!(shadow)

    return (nothing, nothing, nothing)
end

_zero_precision!(A::SparseMatrixCSC) = (fill!(nonzeros(A), 0); A)
_zero_precision!(A::Union{Symmetric, Hermitian}) = (_zero_precision!(A.data); A)
_zero_precision!(A::SymTridiagonal) = (fill!(A.dv, 0); fill!(A.ev, 0); A)
_zero_precision!(A) = (fill!(A, 0); A)
