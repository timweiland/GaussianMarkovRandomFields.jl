# Shared plumbing for the Enzyme rules.
#
# Every rule in this extension routes through the functions below rather than
# reaching for a concrete type's fields. The rules used to be registered on
# `Annotation{<:AbstractGMRF}` while their bodies read `x.val.linsolve_cache` and
# `x.dval.mean` — fields only `GMRF` has — so a `ChordalGMRF` (fields `μ, Q, F`)
# hit a `FieldError` from deep inside the reverse pass. Supporting a new GMRF
# type now means adding methods here; anything without them fails with a message
# that names the type.

"""
    enzyme_selinv(d::AbstractGMRF)

Selected inverse of `d`'s precision matrix, on the factorization's fill pattern.

Each backend hands this back in its own representation — `Symmetric` over full
storage (CHOLMOD), a bare `SparseMatrixCSC` (workspace), or `Hermitian` holding a
single triangle (chordal) — so callers must index the result as a matrix
(`Σ[i, j]`) and never unwrap `.data`, which reads one bare triangle on some
backends and the whole matrix on others.
"""
enzyme_selinv(d::GMRF) = selinv(d.linsolve_cache)
enzyme_selinv(d::WorkspaceGMRF) = (GMRFs.ensure_loaded!(d); selinv(d.workspace))
enzyme_selinv(d::ChordalGMRF) = mselinv(d.Q, d.F)
enzyme_selinv(d::MetaGMRF) = enzyme_selinv(d.gmrf)
enzyme_selinv(d::AbstractGMRF) = enzyme_unsupported("selected inversion", d)

"""
    enzyme_check_supported(op, d::AbstractGMRF)

Reject GMRFs whose *contents* put them outside what a rule handles, as opposed to
their type. Called from every rule's forward pass so the refusal happens before
any work, not halfway through the reverse pass.
"""
enzyme_check_supported(op, d::AbstractGMRF) = nothing
enzyme_check_supported(op, d::MetaGMRF) = enzyme_check_supported(op, d.gmrf)

# A constrained `WorkspaceGMRF` reports its *constrained* mean from `mean(d)`
# while the precision cotangent is defined against the unconstrained one, and
# `logpdf` carries an extra `log_constraint_correction` term that itself depends
# on `Q`. The ChainRules rrule handles both; these rules do not, and quietly
# using the wrong mean would be indistinguishable from a correct answer.
function enzyme_check_supported(op, d::WorkspaceGMRF)
    GMRFs.has_constraints(d) &&
        enzyme_unsupported("$op with linear constraints", d)
    return nothing
end

"""
    shadow_mean(shadow::AbstractGMRF)
    shadow_precision(shadow::AbstractGMRF)

Mean and precision slots of an Enzyme *shadow* GMRF — the zero-initialised
instance Enzyme allocates alongside the primal to accumulate cotangents into.
Field names differ per type (`mean`/`precision` vs `μ`/`Q`), so rules go through
these instead of `getproperty`.
"""
shadow_mean(s::GMRF) = s.mean
shadow_precision(s::GMRF) = s.precision
shadow_mean(s::WorkspaceGMRF) = s.mean
shadow_precision(s::WorkspaceGMRF) = s.precision
shadow_mean(s::ChordalGMRF) = s.μ
shadow_precision(s::ChordalGMRF) = s.Q

# `MetaGMRF` only attaches metadata and forwards every operation, so it inherits
# whatever its inner GMRF supports. (`ConstrainedGMRF` deliberately does *not* get
# this treatment: its `logpdf` carries a constraint correction that depends on `Q`,
# so delegating to the base GMRF would quietly drop a term.)
shadow_mean(s::MetaGMRF) = shadow_mean(s.gmrf)
shadow_precision(s::MetaGMRF) = shadow_precision(s.gmrf)

"""
    precision_storage(d::AbstractGMRF)

How `d`'s stored precision entries map onto the matrix it actually factorizes.

This is a property of the precision's *wrapper*, not of the GMRF type: the same
`GMRF` holds a bare `SparseMatrixCSC` when built from a user's `Q` and a
`Symmetric` one when it came out of `gaussian_approximation`.

- `Val(:full)` — every stored `(i, j)` is read as-is, so each carries its own
  independent cotangent.
- `Val(:lower)` / `Val(:upper)` — only that triangle is read and the other is its
  mirror (`ChordalGMRF` wraps its input in `Hermitian(Q, :L)`; approximation
  posteriors come back as `Symmetric(Q, :U)`). Stored entries in the *ignored*
  triangle then have zero influence on the result, while each off-diagonal entry
  in the stored triangle moves two positions of the effective matrix, so its
  cotangent doubles.

Getting this wrong costs a silent factor of two on off-diagonals, which is why
[`accumulate_on_pattern!`](@ref) dispatches on it rather than assuming.
"""
precision_storage(d::AbstractGMRF) = _storage_convention(precision_matrix(d))

_storage_convention(::AbstractMatrix) = Val(:full)
_storage_convention(A::Union{Symmetric, Hermitian}) =
    A.uplo == 'U' ? Val(:upper) : Val(:lower)

"""
    accumulate_on_pattern!(f, target, storage)

Add `f(i, j)` into every stored entry `(i, j)` of the precision shadow `target`,
**without changing `target`'s sparsity pattern**.

Enzyme's shadow of a `SparseMatrixCSC` is positionally aligned with its primal:
`shadow.nzval[k]` is the cotangent of `primal.nzval[k]`. Precision cotangents are
built from a selected inverse, which lives on the *Cholesky* fill pattern — a
strict superset of `Q`'s, 644 stored entries against `Q`'s 288 for an 8×8 grid
Laplacian. The obvious `target .+= Q̄` therefore runs sparse `broadcast!`, which
grows `target`'s `colptr`/`rowval`/`nzval` in place and desynchronises the shadow
from its primal. The damage surfaces as garbage gradients on Julia 1.10 and as
`ArgumentError: Invalid buffers for SparseMatrixCSC construction` on 1.12, once
something downstream reads the corrupted shadow.

Visiting only `target`'s stored entries also drops the fill-in, which is what the
chain rule asks for: those positions are structural zeros of `Q`, not variables.

`f` is called once per stored entry, so it can close over a selected inverse and
a residual vector without ever materializing the dense `r * r'` outer product.
"""
function accumulate_on_pattern!(f, target::SparseMatrixCSC, ::Val{:full})
    rows = rowvals(target)
    vals = nonzeros(target)
    @inbounds for j in axes(target, 2), p in nzrange(target, j)
        vals[p] += f(rows[p], j)
    end
    return target
end

function accumulate_on_pattern!(f, target::SparseMatrixCSC, ::Val{:lower})
    rows = rowvals(target)
    vals = nonzeros(target)
    @inbounds for j in axes(target, 2), p in nzrange(target, j)
        i = rows[p]
        i < j && continue                       # upper triangle is never read
        vals[p] += (i == j ? f(i, j) : 2 * f(i, j))
    end
    return target
end

function accumulate_on_pattern!(f, target::SparseMatrixCSC, ::Val{:upper})
    rows = rowvals(target)
    vals = nonzeros(target)
    @inbounds for j in axes(target, 2), p in nzrange(target, j)
        i = rows[p]
        i > j && continue                       # lower triangle is never read
        vals[p] += (i == j ? f(i, j) : 2 * f(i, j))
    end
    return target
end

# A triangle-wrapped precision shares its parent's buffers, so writing through
# `.data` lands in the shadow of the caller's original sparse matrix — exactly
# where the cotangent belongs. Spelled out per storage kind rather than left
# generic, which would be ambiguous against the `AbstractMatrix` method below.
for S in (:(Val{:full}), :(Val{:lower}), :(Val{:upper}))
    @eval accumulate_on_pattern!(f, target::Union{Symmetric, Hermitian}, storage::$S) =
        (accumulate_on_pattern!(f, target.data, storage); target)
end

# `SymTridiagonal` and dense precisions have no fill-in problem — their storage
# *is* their pattern — but they still need the parameter-to-entry mapping right:
# `ev[i]` backs both `(i, i+1)` and `(i+1, i)`.
function accumulate_on_pattern!(f, target::SymTridiagonal, ::Val{:full})
    n = length(target.dv)
    @inbounds for i in 1:n
        target.dv[i] += f(i, i)
    end
    @inbounds for i in 1:(n - 1)
        target.ev[i] += f(i + 1, i) + f(i, i + 1)
    end
    return target
end

function accumulate_on_pattern!(f, target::AbstractMatrix, ::Val{:full})
    @inbounds for j in axes(target, 2), i in axes(target, 1)
        target[i, j] += f(i, j)
    end
    return target
end

function accumulate_on_pattern!(f, target::AbstractMatrix, ::Val{:lower})
    @inbounds for j in axes(target, 2), i in axes(target, 1)
        i < j && continue
        target[i, j] += (i == j ? f(i, j) : 2 * f(i, j))
    end
    return target
end

function accumulate_on_pattern!(f, target::AbstractMatrix, ::Val{:upper})
    @inbounds for j in axes(target, 2), i in axes(target, 1)
        i > j && continue
        target[i, j] += (i == j ? f(i, j) : 2 * f(i, j))
    end
    return target
end

"""
    enzyme_unsupported(op, d)

Refuse to differentiate `op` for this GMRF type rather than letting Enzyme walk
into the sparse factorization underneath, where it either crashes or — worse —
returns a plausible-looking wrong number.
"""
@noinline function enzyme_unsupported(op, d)
    name = d isa AbstractGMRF ? string(typeof(d).name.name) : string(typeof(d))
    return throw(
        ArgumentError(
            """
            Enzyme cannot differentiate `$op` for $name.

            Supporting it would mean differentiating through the sparse \
            factorization itself, which produces silently incorrect gradients \
            rather than an error, so this package refuses instead.

            Use Zygote, Mooncake or ForwardDiff for this operation. The \
            per-backend support matrix is in the "Automatic Differentiation" \
            reference page of the documentation.
            """
        )
    )
end
