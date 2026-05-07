# Constrained WorkspaceGMRF with Duals.
#
# The 5-arg constructor WorkspaceGMRF(μ, Q, ws, A, e) builds a ConstraintInfo
# which stores Ã^T = Q⁻¹A' and L_c = chol(A Ã^T) as Float64. For the
# unconstrained Dual flow, A_tilde_T_primal is enough because logpdf uses the
# unconstrained mean and Q. For the constrained flow, log_constraint_correction
# depends on Q through L_c and Ã^T — if we stored only primal values, the
# Q-path derivatives would be silently dropped.
#
# We resolve this by computing A_tilde_T and L_c with full Dual propagation
# (via implicit differentiation of Q Ã^T = A') and using those Dual values
# to form constrained_mean and log_constraint_correction. The struct-stored
# A_tilde_T / L_c stay primal — they're only used for sampling and var, which
# users don't typically differentiate through.

function _compute_constrained_duals(
        mean_T, Q::SparseMatrixCSC{<:ForwardDiff.Dual}, ws::GMRFs.GMRFWorkspace,
        A_dense::Matrix{Float64}, e_vec::Vector{Float64},
        A_tilde_T_v::Matrix{Float64}, log_AA_det::Float64
    )
    D = eltype(Q)
    Tag = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    n = size(Q, 1)
    m = size(A_dense, 1)

    # Build Ã^T with Dual values via implicit diff.
    # Q Ã^T = A' (with A primal) gives, per partial direction k:
    #   Q_v Ã^T_p = -Q_p Ã^T_v
    A_tilde_T_partials = zeros(V, n, m, N)
    for k in 1:N
        Q_p_k_nzval = V[ForwardDiff.partials(Q.nzval[idx], k) for idx in eachindex(Q.nzval)]
        Q_p_k = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, Q_p_k_nzval)
        for i in 1:m
            rhs = -(Q_p_k * @view(A_tilde_T_v[:, i]))
            A_tilde_T_partials[:, i, k] .= GMRFs.workspace_solve(ws, rhs)
        end
    end

    A_tilde_T_dual = Matrix{D}(undef, n, m)
    @inbounds for j in 1:n, i in 1:m
        A_tilde_T_dual[j, i] = ForwardDiff.Dual{Tag, V, N}(
            A_tilde_T_v[j, i],
            ForwardDiff.Partials{N, V}(ntuple(k -> A_tilde_T_partials[j, i, k], N)),
        )
    end

    # Dual L_c via dense Cholesky (m×m, small).
    AAtt_dual = A_dense * A_tilde_T_dual
    L_c_dual = cholesky(Symmetric(AAtt_dual))

    residual = A_dense * mean_T - e_vec
    resid_e = e_vec - A_dense * mean_T
    constrained_mean = mean_T - A_tilde_T_dual * (L_c_dual \ residual)
    log_constraint_correction =
        0.5 * (m * log(2π) + logdet(L_c_dual) + dot(resid_e, L_c_dual \ resid_e)) -
        0.5 * log_AA_det

    return constrained_mean, log_constraint_correction
end

# Kernel: build a constrained Dual WorkspaceGMRF assuming `ws` is already
# loaded (factorized) at Q's primal values, `ws.loaded_version == version`,
# and primal `A_tilde_T_v`, `L_c_primal`, `log_AA_det` are computed against
# that same factorization. Used by both the from-scratch constructor (which
# does the load + solves itself) and the obs-dual workspace path (which
# lifts these from `posterior_primal.constraints`).
function _build_constrained_dual_workspace_gmrf(
        mean::AbstractVector, Q::SparseMatrixCSC, ws::GMRFs.GMRFWorkspace,
        A_dense::Matrix{Float64}, e_vec::Vector{Float64},
        A_tilde_T_v::Matrix{Float64}, L_c_primal,
        log_AA_det::Float64, version::Int
    )
    T = promote_type(eltype(mean), eltype(Q))
    mean_T = convert(Vector{T}, mean)
    Q_T = if eltype(Q) === T
        Q
    else
        SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))
    end
    m = size(A_dense, 1)

    if eltype(Q) <: ForwardDiff.Dual
        constrained_mean, log_constraint_correction = _compute_constrained_duals(
            mean_T, Q, ws, A_dense, e_vec, A_tilde_T_v, log_AA_det
        )
    else
        # Dual-μ-only case: primal Ã^T / L_c are exact; μ-path Dual arithmetic
        # through the trailing `residual` terms is sufficient.
        residual = A_dense * mean_T - e_vec
        resid_e = e_vec - A_dense * mean_T
        constrained_mean = mean_T - A_tilde_T_v * (L_c_primal \ residual)
        log_constraint_correction =
            0.5 * (m * log(2π) + logdet(L_c_primal) + dot(resid_e, L_c_primal \ resid_e)) -
            0.5 * log_AA_det
    end

    ci = GMRFs.ConstraintInfo{T}(
        A_dense, e_vec, A_tilde_T_v, L_c_primal, constrained_mean, log_constraint_correction
    )
    B = typeof(ws.backend)
    return GMRFs.WorkspaceGMRF{T, B, typeof(ws), GMRFs.ConstraintInfo{T}}(
        mean_T, copy(Q_T), ws, ci, version
    )
end

function _construct_forwarddiff_constrained_workspace_gmrf(
        mean::AbstractVector, Q::SparseMatrixCSC, ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix, e::AbstractVector
    )
    GMRFs._check_workspace_pattern(Q, ws)

    # Load primal Q into ws so the primal factorization is current.
    Q_v_nzval = eltype(Q) <: ForwardDiff.Dual ?
        ForwardDiff.value.(Q.nzval) : Vector{Float64}(Q.nzval)
    Q_primal = SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, Q_v_nzval)
    GMRFs.update_precision!(ws, Q_primal)
    version = GMRFs._next_version!(ws)
    ws.loaded_version = version

    n = size(Q, 1)
    m = size(A, 1)
    A_dense = Matrix{Float64}(A)
    e_vec = Vector{Float64}(e)

    # Primal Ã^T = Q_v⁻¹ A' via m solves against the primal factorization.
    A_tilde_T_v = Matrix{Float64}(undef, n, m)
    for i in 1:m
        A_tilde_T_v[:, i] .= GMRFs.workspace_solve(ws, A_dense[i, :])
    end
    L_c_primal = cholesky(Symmetric(A_dense * A_tilde_T_v))
    log_AA_det = logdet(cholesky(Symmetric(A_dense * A_dense')))

    return _build_constrained_dual_workspace_gmrf(
        mean, Q, ws, A_dense, e_vec, A_tilde_T_v, L_c_primal, log_AA_det, version
    )
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC,
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector,
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end

function GMRFs.WorkspaceGMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        Q::SparseMatrixCSC{<:ForwardDiff.Dual},
        ws::GMRFs.GMRFWorkspace,
        A::AbstractMatrix,
        e::AbstractVector
    )
    return _construct_forwarddiff_constrained_workspace_gmrf(mean, Q, ws, A, e)
end
