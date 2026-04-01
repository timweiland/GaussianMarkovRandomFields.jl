import CliqueTrees
import CliqueTrees.Multifrontal: ChordalCholesky, selinv!, triangular

export CliqueTreesBackend

"""
    CliqueTreesBackend{T, I, F} <: WorkspaceBackend

Pure-Julia sparse Cholesky backend via CliqueTrees.jl. Thread-safe — no global
state, so multiple backends can be used concurrently from different tasks.

Stores two `ChordalCholesky` objects sharing the same symbolic factorization:
- `factor`: holds the numeric Cholesky factor (for solve, logdet, backward solve)
- `selinv_factor`: holds the selected inverse after `compute_selinv!`

The selinv result stays cached in `selinv_factor` until the next `refactorize!`.

Also stores a lower-triangle buffer (`Q_lower`) and a precomputed index map
from the full symmetric Q's nzval to Q_lower's nzval for zero-scan updates.
"""
struct CliqueTreesBackend{T, I, F <: ChordalCholesky} <: WorkspaceBackend
    factor::F
    selinv_factor::F
    Q_lower::SparseMatrixCSC{T, I}
    full_to_lower::Vector{Int}
end

function CliqueTreesBackend(Q::SparseMatrixCSC{T}; alg = CliqueTrees.AMD()) where {T}
    Q_lower = sparse(tril(Q))
    full_to_lower = _build_full_to_lower_map(Q, Q_lower)

    factor = ChordalCholesky(Q_lower; alg = alg)
    copyto!(factor, Q_lower)
    cholesky!(factor)

    selinv_factor = deepcopy(factor)

    return CliqueTreesBackend{T, Int, typeof(factor)}(
        factor, selinv_factor, Q_lower, full_to_lower
    )
end

function refactorize!(b::CliqueTreesBackend, Q::Symmetric)
    _update_lower_from_full!(b.Q_lower, Q.data, b.full_to_lower)
    copyto!(b.factor, b.Q_lower)
    cholesky!(b.factor)
    return nothing
end

function backend_solve(b::CliqueTreesBackend, rhs::AbstractVector)
    return b.factor \ Vector(rhs)
end

function compute_logdet(b::CliqueTreesBackend)
    return logdet(b.factor)
end

function compute_selinv!(b::CliqueTreesBackend)
    _copy_factor!(b.selinv_factor, b.factor)
    selinv!(b.selinv_factor)
    return nothing
end

function get_selinv(b::CliqueTreesBackend{T}) where {T}
    # Symmetrize + unpermute from the ChordalCholesky storage
    L_sel = sparse(triangular(b.selinv_factor))
    invp = collect(b.selinv_factor.invp)
    Σ_perm = L_sel + L_sel' - Diagonal(diag(L_sel))
    return Σ_perm[invp, invp]
end

function get_selinv_diag(b::CliqueTreesBackend)
    L_sel = sparse(triangular(b.selinv_factor))
    invp = collect(b.selinv_factor.invp)
    return diag(L_sel)[invp]
end

function backend_backward_solve(b::CliqueTreesBackend, x::AbstractVector)
    return b.factor' \ Vector(x)
end

# --- Helpers ---

function _copy_factor!(dst::ChordalCholesky, src::ChordalCholesky)
    copyto!(dst.Dval, src.Dval)
    copyto!(dst.Lval, src.Lval)
    return nothing
end

function _build_full_to_lower_map(Q_full::SparseMatrixCSC, Q_lower::SparseMatrixCSC)
    fmap = zeros(Int, nnz(Q_full))
    Q_full_rows = rowvals(Q_full)
    Q_lower_rows = rowvals(Q_lower)
    for col in 1:size(Q_full, 2)
        lower_ptr = first(nzrange(Q_lower, col))
        lower_end = last(nzrange(Q_lower, col))
        for full_idx in nzrange(Q_full, col)
            full_row = Q_full_rows[full_idx]
            if full_row < col
                continue
            end
            while lower_ptr <= lower_end && Q_lower_rows[lower_ptr] < full_row
                lower_ptr += 1
            end
            if lower_ptr <= lower_end && Q_lower_rows[lower_ptr] == full_row
                fmap[full_idx] = lower_ptr
            end
        end
    end
    return fmap
end

function _update_lower_from_full!(
        Q_lower::SparseMatrixCSC, Q_full::SparseMatrixCSC, full_to_lower::Vector{Int}
    )
    @inbounds for k in eachindex(full_to_lower)
        idx = full_to_lower[k]
        if idx > 0
            Q_lower.nzval[idx] = Q_full.nzval[k]
        end
    end
    return nothing
end

"""
    GMRFWorkspace(Q::SparseMatrixCSC, ::Type{CliqueTreesBackend}; alg=CliqueTrees.AMD())

Create a workspace using the CliqueTrees backend (pure Julia, thread-safe).
The `alg` keyword selects the fill-reducing ordering algorithm (e.g., `CliqueTrees.AMD()`,
`CliqueTrees.METIS()`, `MMD()`).
"""
function GMRFWorkspace(Q::SparseMatrixCSC{T}, ::Type{CliqueTreesBackend}; alg = CliqueTrees.AMD()) where {T}
    n = size(Q, 1)
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))

    backend = CliqueTreesBackend(Q; alg = alg)

    return GMRFWorkspace{T, typeof(backend)}(
        copy(Q),
        backend,
        zeros(T, n),
        zeros(T, n),
        true,
        false,
        false,
        zero(T),
        1,  # next_version
        0,  # loaded_version
    )
end
