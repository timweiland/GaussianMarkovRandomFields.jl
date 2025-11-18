using LinearAlgebra, LinearSolve, SparseArrays

export sparse_approximate_cholesky!, sparse_approximate_cholesky, approximate_gmrf_kl

"""
    sparse_approximate_cholesky!(Θ::AbstractMatrix, L::SparseMatrixCSC)

In-place computation of sparse approximate Cholesky factorization.

Fill the nonzero values of the lower-triangular sparse matrix `L` such that
`L * L' ≈ Θ⁻¹`, where `Θ` is a covariance matrix and `L * L'` is the approximate
precision (inverse covariance) matrix.

This function uses the sparsity pattern already defined in `L` and fills in the
values using local Cholesky decompositions at each step. The approximation quality
depends on the sparsity pattern of `L`.

# Arguments
- `Θ::AbstractMatrix`: Covariance matrix to be inverted approximately
- `L::SparseMatrixCSC`: Sparse lower-triangular matrix with predefined sparsity pattern.
  Values will be overwritten.

# Details
The algorithm proceeds column-by-column through `L`, solving local linear systems
using dense Cholesky factorizations on small submatrices of `Θ`. A small regularization
term (`1e-6 * I`) is added for numerical stability.

# See also
[`sparse_approximate_cholesky`](@ref), [`approximate_gmrf_kl`](@ref)
"""
function sparse_approximate_cholesky!(Θ::AbstractMatrix, L::SparseMatrixCSC)
    max_buffer_size = maximum(length(rowvals(L)[nzrange(L, k)]) for k in 1:size(L, 2))
    cho_buf = Vector{Float64}(undef, max_buffer_size^2)
    x_buf = Vector{Float64}(undef, max_buffer_size)
    for k in 1:size(L, 2)
        nz_idcs = reverse(nzrange(L, k))
        S = rowvals(L)[nz_idcs]
        Nₛ = length(S)
        local_cho_buf = reshape(view(cho_buf, 1:(Nₛ^2)), Nₛ, Nₛ)
        local_cho_buf .= Θ[S, S] + 1.0e-6 * I
        #println(minimum(eigvals(Array(local_cho_buf))))
        Θₛ_cho = cholesky!(local_cho_buf)
        local_x_buf = view(x_buf, 1:Nₛ)
        fill!(local_x_buf, 0.0)
        local_x_buf[end] = 1.0
        ldiv!(Θₛ_cho.U, local_x_buf)
        L.nzval[nz_idcs] .= local_x_buf
    end
    return
end

function _build_supernodal_sparsity_pattern(sc::GaussianMarkovRandomFields.SupernodeClustering)
    Is = Int[]
    Js = Int[]
    for s in 1:length(sc)
        for j in sc.column_indices[s], i in sc.row_indices[s]
            if j <= i
                push!(Is, i)
                push!(Js, j)
            end
        end
    end
    return spzeros(Float64, Is, Js)
end

function sparse_approximate_cholesky(Θ::AbstractMatrix, sc::SupernodeClustering)
    max_pattern_len = mapreduce(length, max, sc.row_indices)
    max_column_len = mapreduce(length, max, sc.column_indices)
    cho_buf = Vector{Float64}(undef, max_pattern_len^2)
    x_buf = Vector{Float64}(undef, max_pattern_len * max_column_len)
    pattern_buf = Vector{Int}(undef, max_pattern_len)

    L = _build_supernodal_sparsity_pattern(sc)

    N_sn = length(sc)
    for s in 1:N_sn
        Nₛ = length(sc.row_indices[s])
        local_pattern_buf = view(pattern_buf, 1:Nₛ)
        copyto!(pattern_buf, sc.row_indices[s])

        parents = sc.column_indices[s]
        N_parents = length(parents)

        local_cho_buf = reshape(view(cho_buf, 1:(Nₛ^2)), Nₛ, Nₛ)
        local_cho_buf .= Θ[local_pattern_buf, local_pattern_buf] + 1.0e-8 * I
        Θₛ_cho = cholesky!(local_cho_buf)
        local_x_buf = reshape(view(x_buf, 1:(Nₛ * N_parents)), Nₛ, N_parents)
        fill!(local_x_buf, 0.0)
        for k in 1:N_parents
            N_k = length(nzrange(L, parents[k]))
            local_x_buf[N_k, k] = 1.0
        end
        ldiv!(Θₛ_cho.U, local_x_buf)

        for k in 1:N_parents
            cur_nzrange = nzrange(L, parents[k])
            N_k = length(cur_nzrange)
            L.nzval[cur_nzrange] .= local_x_buf[N_k:-1:1, k]
        end
    end
    return L
end

"""
    sparse_approximate_cholesky(Θ::AbstractMatrix, X::AbstractMatrix; ρ = 2.0, λ = 1.5)

Compute sparse approximate Cholesky factorization with automatic sparsity pattern.

Given a covariance matrix `Θ` and spatial input points `X`, compute a sparse approximate
Cholesky factor `L` and permutation `P` such that `L * L' ≈ (Θ[P, P])⁻¹`.

# Arguments
- `Θ::AbstractMatrix`: Covariance matrix to be inverted approximately
- `X::AbstractMatrix`: Input point locations (d × n matrix where d is spatial dimension, n is number of points)

# Keyword Arguments
- `ρ::Real = 2.0`: Sparsity pattern radius parameter. Controls the neighborhood size used
  to determine sparsity. Larger values create denser (more accurate) approximations.
  Typical values are in the range [1.5, 3.0].
- `λ::Union{Real,Nothing} = 1.5`: Supernodal clustering parameter. If `nothing`, uses
  standard column-by-column factorization. If a number, uses supernodal clustering with the
  given threshold (typically 1.5). Supernodal factorization groups nearby columns together
  for improved cache efficiency and is usually faster.

# Returns
- `L::SparseMatrixCSC`: Lower-triangular sparse Cholesky factor in permuted coordinates
- `P::Vector{Int}`: Permutation vector from the reverse maximin ordering

# Details
The permutation `P` reorders the points to achieve better sparsity. The returned factor
`L` satisfies `L * L' ≈ (Θ[P, P])⁻¹`, or equivalently `(L * L')[invperm(P), invperm(P)] ≈ Θ⁻¹`.

When `λ !== nothing`, the algorithm uses supernodal clustering to group columns with
similar sparsity patterns, solving larger dense systems less frequently rather than many
small systems. This typically improves performance through better cache locality.

# See also
[`sparse_approximate_cholesky!`](@ref), [`approximate_gmrf_kl`](@ref), [`reverse_maximin_ordering`](@ref)
"""
function sparse_approximate_cholesky(Θ::AbstractMatrix, X::AbstractMatrix; ρ = 2.0, λ = 1.5)
    if λ === nothing
        # Non-supernodal version
        L, P, _ = reverse_maximin_ordering_and_sparsity_pattern(X, ρ)
        Θ_P = PermutedMatrix(Θ, P)
        sparse_approximate_cholesky!(Θ_P, L)
        return L, P
    else
        # Supernodal version
        S, P, ℓ = reverse_maximin_ordering_and_sparsity_pattern(X, ρ)
        sc = form_supernodes(S, P, ℓ; λ)
        Θ_P = PermutedMatrix(Θ, P)
        return sparse_approximate_cholesky(Θ_P, sc), P
    end
end

"""
    approximate_gmrf_kl(kernel_mat::AbstractMatrix, X::AbstractMatrix; ρ = 2.0, λ = 1.5, alg = LinearSolve.CHOLMODFactorization())

Construct a sparse GMRF approximation from a kernel (covariance) matrix.

Given a kernel matrix `K` (covariance matrix) obtained by evaluating a kernel function
at input points `X`, construct a sparse Gaussian Markov Random Field (GMRF) that
approximates the corresponding Gaussian process.

# Arguments
- `kernel_mat::AbstractMatrix`: Kernel (covariance) matrix
- `X::AbstractMatrix`: Input point locations (d × n matrix where d is spatial dimension, n is number of points)

# Keyword Arguments
- `ρ::Real = 2.0`: Sparsity pattern radius parameter. Larger values create denser, more accurate approximations. Typical range: [1.5, 3.0].
- `λ::Union{Real,Nothing} = 1.5`: Supernodal clustering parameter. If `nothing`, uses
  standard column-by-column factorization. If a number, uses supernodal clustering (typically 1.5).
  Supernodal factorization is usually faster due to improved cache efficiency.
- `alg = LinearSolve.CHOLMODFactorization()`: LinearSolve algorithm for the resulting GMRF

# Returns
- `::GMRF`: A sparse GMRF with zero mean and sparse precision matrix approximating `K⁻¹`

# Examples
```julia
using KernelFunctions, GaussianMarkovRandomFields

# Create spatial grid
X = hcat([[x, y] for x in 0:0.1:1, y in 0:0.1:1]...)

# Define kernel and compute kernel matrix
kernel = with_lengthscale(Matern32Kernel(), 0.2)
K = kernelmatrix(kernel, X, obsdim=2)

# Create sparse GMRF approximation (uses supernodal by default)
gmrf = approximate_gmrf_kl(K, X; ρ=2.0)

# Use non-supernodal version
gmrf_nonsupernodal = approximate_gmrf_kl(K, X; ρ=2.0, λ=nothing)

# Use for inference
posterior = linear_condition(gmrf; A=A, Q_ϵ=Q_ϵ, y=y)
```

# See also
[`sparse_approximate_cholesky`](@ref), [`GMRF`](@ref), [`linear_condition`](@ref)
"""
function approximate_gmrf_kl(kernel_mat::AbstractMatrix, X::AbstractMatrix; ρ = 2.0, λ = 1.5, alg = LinearSolve.CHOLMODFactorization())
    L, P = sparse_approximate_cholesky(kernel_mat, X; ρ, λ)
    P_inv = invperm(P)
    Q = (L * L')[P_inv, P_inv]
    N = size(Q, 2)
    return GMRF(zeros(N), Q, alg)
end
