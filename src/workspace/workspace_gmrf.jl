import Distributions: logpdf

export WorkspaceGMRF, ConstraintInfo

"""
    ConstraintInfo{T}

Precomputed constraint quantities for a constrained WorkspaceGMRF.
Stores the constraint matrix A, vector e, and derived quantities
(Ã^T = Q⁻¹A^T, L_c = chol(AÃ^T), constrained mean, log correction).
"""
struct ConstraintInfo{T}
    matrix::Matrix{Float64}
    vector::Vector{Float64}
    A_tilde_T::Matrix{Float64}
    L_c::Cholesky{Float64, Matrix{Float64}}
    constrained_mean::Vector{T}
    log_constraint_correction::T
end

function ConstraintInfo(
        ws::GMRFWorkspace, μ::AbstractVector{T},
        A::AbstractMatrix, e::AbstractVector
    ) where {T}
    n = dimension(ws)
    m, n_A = size(A)
    n == n_A ||
        throw(ArgumentError("Constraint matrix size $(size(A)) incompatible with workspace size $(n)"))
    m == length(e) ||
        throw(ArgumentError("Constraint matrix rows $(m) != constraint vector length $(length(e))"))

    A_dense = Matrix{Float64}(A)
    e_vec = Vector{Float64}(e)

    # Ã^T = Q⁻¹A^T via m column solves
    A_tilde_T = Matrix{Float64}(undef, n, m)
    for i in 1:m
        A_tilde_T[:, i] .= workspace_solve(ws, A_dense[i, :])
    end

    # AÃ^T and its Cholesky
    L_c = cholesky(Symmetric(A_dense * A_tilde_T))

    # Constrained mean
    residual = A_dense * μ - e_vec
    constrained_mean = μ - A_tilde_T * (L_c \ residual)

    # Log-density correction (Rue & Held 2005, §2.3.3)
    resid_e = e_vec - A_dense * μ
    r = length(resid_e)
    log_constraint_correction =
        0.5 * (r * log(2π) + logdet(L_c) + dot(resid_e, L_c \ resid_e)) -
        0.5 * logdet(cholesky(Symmetric(A_dense * A_dense')))

    return ConstraintInfo{T}(
        A_dense, e_vec, A_tilde_T, L_c, constrained_mean, log_constraint_correction
    )
end

"""
    WorkspaceGMRF{T, B, W, C} <: AbstractGMRF{T, SparseMatrixCSC{T, Int}}

A GMRF backed by a `GMRFWorkspace` for high-performance repeated operations,
with optional linear equality constraints.

Each `WorkspaceGMRF` owns a snapshot of its precision matrix (`nzval` copy)
and a `version` tag. Before any workspace operation, `ensure_loaded!` checks
whether the workspace currently holds this GMRF's data — if not, it reloads
the precision values and triggers refactorization. This makes it safe for
multiple `WorkspaceGMRF`s to share the same workspace (at the cost of a
refactorization when switching between them).

# Fields
- `mean`: The unconstrained mean vector.
- `precision`: This GMRF's precision matrix (snapshot, owned by this instance).
- `workspace`: Shared factorization engine.
- `constraints`: `nothing` or `ConstraintInfo{T}` with precomputed constraint quantities.
- `version`: Workspace version tag — used by `ensure_loaded!` to detect stale state.
"""
struct WorkspaceGMRF{
        T <: Real, B <: WorkspaceBackend, W <: GMRFWorkspace,
        C <: Union{Nothing, ConstraintInfo{T}},
    } <: AbstractGMRF{T, SparseMatrixCSC{T, Int}}
    mean::Vector{T}
    precision::SparseMatrixCSC{T, Int}
    workspace::W
    constraints::C
    version::Int
end

# --- Constructors ---

"""
    WorkspaceGMRF(mean, Q::SparseMatrixCSC)

Create an unconstrained `WorkspaceGMRF` with a new workspace.
"""
function WorkspaceGMRF(mean::AbstractVector{T}, Q::SparseMatrixCSC{T}) where {T}
    ws = GMRFWorkspace(Q)
    version = _next_version!(ws)
    ws.loaded_version = version
    return WorkspaceGMRF{T, typeof(ws.backend), typeof(ws), Nothing}(
        Vector{T}(mean), copy(Q), ws, nothing, version
    )
end

"""
    WorkspaceGMRF(mean, Q::SparseMatrixCSC, ws::GMRFWorkspace)

Create an unconstrained `WorkspaceGMRF` reusing an existing workspace.
"""
function WorkspaceGMRF(
        mean::AbstractVector{T}, Q::SparseMatrixCSC{T}, ws::GMRFWorkspace
    ) where {T}
    version = _next_version!(ws)
    B = typeof(ws.backend)
    return WorkspaceGMRF{T, B, typeof(ws), Nothing}(
        Vector{T}(mean), copy(Q), ws, nothing, version
    )
end

"""
    WorkspaceGMRF(mean, Q::SparseMatrixCSC, ws::GMRFWorkspace, A, e)

Create a constrained `WorkspaceGMRF` with constraints Ax = e.
"""
function WorkspaceGMRF(
        mean::AbstractVector{T}, Q::SparseMatrixCSC{T}, ws::GMRFWorkspace,
        A::AbstractMatrix, e::AbstractVector
    ) where {T}
    version = _next_version!(ws)
    # ensure_loaded! so ConstraintInfo solves use the right Q
    copyto!(ws.Q.nzval, Q.nzval)
    _invalidate!(ws)
    ws.loaded_version = version
    ci = ConstraintInfo(ws, mean, A, e)
    B = typeof(ws.backend)
    return WorkspaceGMRF{T, B, typeof(ws), ConstraintInfo{T}}(
        Vector{T}(mean), copy(Q), ws, ci, version
    )
end

"""
    _next_version!(ws::GMRFWorkspace) -> Int

Assign and return a unique version number from the workspace.
"""
function _next_version!(ws::GMRFWorkspace)
    v = ws.next_version
    ws.next_version += 1
    return v
end

"""
    ensure_loaded!(d::WorkspaceGMRF)

Ensure the workspace currently holds this GMRF's precision data. If the
workspace has been used by a different `WorkspaceGMRF` (or by a direct
`update_precision!` call) since this GMRF was last active, reload and
invalidate the factorization.

!!! warning "Single-task only"
    The version check, `nzval` copy, and `_invalidate!` here are not atomic.
    Sharing one `GMRFWorkspace` between `WorkspaceGMRF`s on multiple concurrent
    tasks is unsafe — a context switch between the version check and the
    refactorization can leave the workspace in an inconsistent state. For
    parallel use, give each task its own workspace (e.g. via `WorkspacePool`).
"""
function ensure_loaded!(d::WorkspaceGMRF)
    ws = d.workspace
    if ws.loaded_version != d.version
        copyto!(ws.Q.nzval, d.precision.nzval)
        _invalidate!(ws)
        ws.loaded_version = d.version
    end
    return nothing
end

has_constraints(d::WorkspaceGMRF) = d.constraints !== nothing

# --- AbstractGMRF interface ---

Base.length(d::WorkspaceGMRF) = size(d.precision, 1)
precision_map(d::WorkspaceGMRF) = d.precision
precision_matrix(d::WorkspaceGMRF) = d.precision

function mean(d::WorkspaceGMRF)
    return d.constraints === nothing ? d.mean : d.constraints.constrained_mean
end

function logdetcov(d::WorkspaceGMRF)
    ensure_loaded!(d)
    return logdet_cov(d.workspace)
end

function var(d::WorkspaceGMRF{T}) where {T}
    ensure_loaded!(d)
    σ_base = selinv_diag(d.workspace)
    if d.constraints === nothing
        return σ_base
    else
        ci = d.constraints
        B_T = ci.L_c.L \ ci.A_tilde_T'
        B_squared_rowsums = vec(sum(abs2, B_T, dims = 1))
        σ_constrained = σ_base - B_squared_rowsums
        σ_constrained .= max.(σ_constrained, zero(T))
        return σ_constrained
    end
end

function _rand!(rng::AbstractRNG, d::WorkspaceGMRF, x::AbstractVector)
    ensure_loaded!(d)
    randn!(rng, x)
    x .= backward_solve(d.workspace, x)
    x .+= d.mean
    if d.constraints !== nothing
        ci = d.constraints
        residual = ci.matrix * Vector(x) - ci.vector
        x .-= ci.A_tilde_T * (ci.L_c \ residual)
    end
    return x
end

function logpdf(d::WorkspaceGMRF, z::AbstractVector)
    ensure_loaded!(d)
    r = z - d.mean
    n = length(d)
    val = -0.5 * dot(r, d.precision * r) - 0.5 * logdetcov(d) - 0.5 * n * log(2π)
    if d.constraints !== nothing
        ci = d.constraints
        # Constraint check
        constraint_residual = ci.matrix * z - ci.vector
        rel_error = norm(constraint_residual) /
            (norm(ci.matrix, Inf) * norm(z, Inf) + 1)
        if rel_error > sqrt(eps())
            @warn "Point does not satisfy constraints (relative residual: $(rel_error))" maxlog = 1
        end
        val += ci.log_constraint_correction
    end
    return val
end

# --- Display ---

function Base.show(io::IO, d::WorkspaceGMRF{T}) where {T}
    c_str = has_constraints(d) ? ", constraints=$(size(d.constraints.matrix, 1))" : ""
    return print(io, "WorkspaceGMRF{$T}(n=$(length(d))$(c_str))")
end

# COV_EXCL_START
function Base.show(io::IO, ::MIME"text/plain", d::WorkspaceGMRF{T}) where {T}
    n = length(d)
    if has_constraints(d)
        m = size(d.constraints.matrix, 1)
        println(io, "WorkspaceGMRF{$T} with $n variables and $m constraint$(m > 1 ? "s" : "")")
    else
        println(io, "WorkspaceGMRF{$T} with $n variables")
    end
    μ = mean(d)
    if length(μ) <= 6
        println(io, "  Mean: $μ")
    else
        println(io, "  Mean: [$(μ[1]), $(μ[2]), $(μ[3]), ..., $(μ[end - 2]), $(μ[end - 1]), $(μ[end])]")
    end
    return print(io, "  Backend: $(typeof(d.workspace.backend))")
end
# COV_EXCL_STOP
