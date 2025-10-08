using SparseArrays
using LinearAlgebra
using ChainRulesCore

# TODO: Make PR for SymTridiagonal rrules into ChainRules, to avoid type piracy
"""
    ChainRulesCore.rrule(::typeof(*), Q::SymTridiagonal, v::AbstractVector)

Custom rrule for SymTridiagonal matrix-vector multiplication.

Zygote's default rrule only accounts for the upper triangle, but each off-diagonal
element appears in both [i,i+1] and [i+1,i], so we need to sum both contributions.
"""
function ChainRulesCore.rrule(
        ::typeof(*),
        Q::SymTridiagonal{T, V},
        v::AbstractVector{<:Union{Real, Complex}}
    ) where {T <: Union{Real, Complex}, V <: AbstractVector{T}}
    y = Q * v

    project_v = ProjectTo(v)
    function symtri_mul_pullback(ȳ)
        # Gradient w.r.t. Q: ȳ ⊗ vᵀ (outer product)
        n = length(v)

        Q̄ = @thunk(
            Tridiagonal(
                ȳ[2:n] .* v[1:(n - 1)],
                ȳ .* v,
                ȳ[1:(n - 1)] .* v[2:n]
            )
        )
        return NoTangent(), Q̄, @thunk(project_v(Q' * ȳ))
    end

    return y, symtri_mul_pullback
end

"""
    ChainRulesCore.rrule(::Type{SymTridiagonal}, dv::AbstractVector, ev::AbstractVector)

ChainRule for SymTridiagonal constructor to enable Zygote differentiation.

The SymTridiagonal stores diagonal (`dv`) and off-diagonal (`ev`) separately.
The pullback extracts these from the incoming tangent matrix.
"""
function ChainRulesCore.rrule(::Type{SymTridiagonal}, dv::AbstractVector, ev::AbstractVector)
    y = SymTridiagonal(dv, ev)
    project_d = ProjectTo(dv)
    project_e = ProjectTo(ev)
    function pullback(ȳ)
        ȳ = unthunk(ȳ)

        # If we’re given a SymTridiagonal as the cotangent, just read its fields.
        if ȳ isa SymTridiagonal
            dd = ȳ.dv
            de = ȳ.ev
        elseif ȳ isa Tangent
            dd = haskey(ȳ, :dv) ? ȳ.dv : ZeroTangent()
            de = haskey(ȳ, :ev) ? ȳ.ev : ZeroTangent()
        elseif ȳ isa AbstractMatrix
            # For off-diagonals, the parameter e[i] contributes to (i,i+1) and (i+1,i),
            # so the adjoint is the sum of those two entries.
            dd = diag(ȳ)
            de = diag(ȳ, 1) + diag(ȳ, -1)
        end

        return (NoTangent(), project_d(dd), project_e(de))
    end
    return y, pullback
end

"""
    compute_precision_gradient(Qinv::AbstractMatrix, r::AbstractVector, ȳ::Real)

Compute the gradient of log-density w.r.t. precision matrix Q.

The gradient is: ∂logpdf/∂Q = 0.5 * ȳ * (Q⁻¹ - r*rᵀ)

This function uses multiple dispatch to efficiently compute the gradient for different
matrix types that may be returned by `selinv`:
- `SparseMatrixCSC`: Uses sparsity pattern to avoid dense operations
- `SymTridiagonal`: Uses tridiagonal structure
- `Symmetric{SparseMatrixCSC}`: Preserves symmetry and sparsity
- Generic fallback: May be inefficient for large matrices (issues warning)

# Arguments
- `Qinv`: Inverse precision matrix from selected inversion
- `r`: Residual vector (z - μ)
- `ȳ`: Incoming gradient scalar

# Returns
Gradient matrix with same structure as `Qinv`
"""
function compute_precision_gradient(Qinv::AbstractMatrix, r::AbstractVector, ȳ::Real)
    @warn "Using generic fallback for precision gradient computation with matrix type $(typeof(Qinv)). " *
        "This may be inefficient for large matrices. Consider using a factorization that returns " *
        "SparseMatrixCSC or SymTridiagonal." maxlog = 1

    # Generic approach - works but allocates full outer product
    return @. 0.5 * ȳ * (Qinv - r * r')
end

"""
    compute_precision_gradient(Qinv::SparseMatrixCSC, r::AbstractVector, ȳ::Real)

Efficient gradient computation for sparse matrices using sparsity pattern.
"""
function compute_precision_gradient(Qinv::SparseMatrixCSC, r::AbstractVector, ȳ::Real)
    # Extract sparsity structure
    rows, cols, vals = findnz(Qinv)

    # Compute outer product values only at nonzero locations
    rr_vals = r[rows] .* r[cols]

    # Build sparse gradient matrix
    return sparse(rows, cols, (0.5 * ȳ) .* (vals .- rr_vals), size(Qinv)...)
end

"""
    compute_precision_gradient(Qinv::SymTridiagonal, r::AbstractVector, ȳ::Real)

Efficient gradient computation for symmetric tridiagonal matrices.
"""
function compute_precision_gradient(Qinv::SymTridiagonal, r::AbstractVector, ȳ::Real)
    n = length(r)

    # Diagonal: Qinv.dv - r .* r
    dv = @. 0.5 * ȳ * (Qinv.dv - r * r)

    # Off-diagonal: Qinv.ev - r[1:n-1] .* r[2:n]
    ev = @. 0.5 * ȳ * (Qinv.ev - r[1:(n - 1)] * r[2:n])

    return SymTridiagonal(dv, ev)
end

"""
    compute_precision_gradient(Qinv::Symmetric{T, <:SparseMatrixCSC}, r, ȳ) where T

Efficient gradient computation for symmetric sparse matrices.
"""
function compute_precision_gradient(Qinv::Symmetric{T, <:SparseMatrixCSC}, r::AbstractVector, ȳ::Real) where {T}
    # Extract sparsity structure from underlying data
    rows, cols, vals = findnz(Qinv.data)

    # Compute outer product values only at nonzero locations
    rr_vals = r[rows] .* r[cols]

    # Build sparse gradient matrix and wrap in Symmetric
    grad_data = sparse(rows, cols, (0.5 * ȳ) .* (vals .- rr_vals), size(Qinv)...)
    return Symmetric(grad_data, Symbol(Qinv.uplo))
end
