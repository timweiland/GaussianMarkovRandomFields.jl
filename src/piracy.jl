#                                  |    |    |
#                                 )_)  )_)  )_)
#                                )___))___))___)
#                               )____)____)_____)
#                             _____|____|____|____
#                    ---------\                   /---------
#                      ^^^^^ ^^^^^^^^^^^^^^^^^^^^^
#                        ^^^^      ^^^^     ^^^    ^^
#                               ^^^^      ^^^
#
# Type piracy to enable autodiff for Hermitian/Symmetric sparse matrices.
# These changes have been submitted as PRs to ChainRulesCore, ChainRules, and Zygote.
# This file can be removed once those PRs are merged and released.

using ChainRulesCore
using ChainRulesCore: ProjectTo, project_type, _projection_mismatch, NoTangent, ZeroTangent, AbstractZero, @thunk, unthunk
using LinearAlgebra
using LinearAlgebra: Hermitian, Symmetric, Adjoint, Transpose, AdjOrTrans, dot, rmul!, tril, triu
using SparseArrays
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, getcolptr, nonzeros

#####
##### Type aliases
#####

const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}

const DenseMat{T} = Union{StridedMatrix{T}, AdjOrTrans{T, <:StridedVecOrMat{T}}}
const DenseVecOrMat{T} = Union{DenseMat{T}, StridedVector{T}}

#####
##### ChainRulesCore: ProjectTo for HermOrSymSparse
#####

const SparseProjectToData{T, I} = NamedTuple{
    (:element, :axes, :rowval, :nzranges, :colptr),
    Tuple{
        ProjectTo{T, NamedTuple{(), Tuple{}}},
        Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},
        Vector{I},
        Vector{UnitRange{Int64}},
        Vector{I},
    },
}

const SparseProjectTo{T, I} = ProjectTo{SparseMatrixCSC, SparseProjectToData{T, I}}

const HermSparseProjectTo{T, I} = ProjectTo{
    Hermitian,
    NamedTuple{
        (:uplo, :parent),
        Tuple{Symbol, SparseProjectTo{T, I}},
    },
}

const SymSparseProjectTo{T, I} = ProjectTo{
    Symmetric,
    NamedTuple{
        (:uplo, :parent),
        Tuple{Symbol, SparseProjectTo{T, I}},
    },
}

function ChainRulesCore.ProjectTo(x::HermSparse{T}) where {T<:Number}
    return ProjectTo{Hermitian}(;
        uplo=Symbol(x.uplo),
        parent=ProjectTo(parent(x)),
    )
end

function ChainRulesCore.ProjectTo(x::SymSparse{T}) where {T<:Number}
    return ProjectTo{Symmetric}(;
        uplo=Symbol(x.uplo),
        parent=ProjectTo(parent(x)),
    )
end

function project!(A::SparseMatrixCSC{T, I}, B::SparseMatrixCSC{<:Any, J}, uplo::Char) where {T, I, J}
    @assert size(A) == size(B)

    @inbounds for j in axes(A, 2)
        p = getcolptr(A)[j]
        pstop = getcolptr(A)[j + 1]
        q = getcolptr(B)[j]
        qstop = getcolptr(B)[j + 1]

        while p < pstop
            i = rowvals(A)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                while q < qstop && rowvals(B)[q] < i
                    q += one(J)
                end

                if q < qstop && rowvals(B)[q] == i
                    nonzeros(A)[p] = nonzeros(B)[q]
                else
                    nonzeros(A)[p] = zero(T)
                end
            end

            p += one(I)
        end
    end

    return A
end

function project!(A::HermOrSymSparse, B::HermOrSymSparse)
    if A.uplo == B.uplo
        project!(parent(A), parent(B), A.uplo)
    elseif A.uplo == 'L'
        project!(parent(A), tril(B), A.uplo)
    else
        project!(parent(A), triu(B), A.uplo)
    end

    return A
end

function sparse_from_project(P::SparseProjectTo{T, I}) where {T, I}
    m, n = map(length, P.axes)
    return SparseMatrixCSC(m, n, P.colptr, P.rowval, zeros(T, length(P.rowval)))
end

function sparse_from_project(P::HermSparseProjectTo)
    return Hermitian(sparse_from_project(P.parent), P.uplo)
end

function sparse_from_project(P::SymSparseProjectTo)
    return Symmetric(sparse_from_project(P.parent), P.uplo)
end

function checkpatternsym(n, Acolptr::Vector{IA}, Bcolptr::Vector{IB}, Arowval::AbstractVector, Browval::AbstractVector, uplo::Char) where {IA, IB}
    for j in 1:n
        pa = Acolptr[j]
        pb = Bcolptr[j]
        pastop = Acolptr[j + 1]
        pbstop = Bcolptr[j + 1]

        while pa < pastop && pb < pbstop
            ia = Arowval[pa]
            ib = Browval[pb]

            if (uplo == 'L' && ia < j) || (uplo == 'U' && ia > j)
                pa += one(IA)
            elseif (uplo == 'L' && ib < j) || (uplo == 'U' && ib > j)
                pb += one(IB)
            elseif ia == ib
                pa += one(IA)
                pb += one(IB)
            else
                return false
            end
        end

        while pa < pastop
            ia = Arowval[pa]

            if (uplo == 'L' && ia >= j) || (uplo == 'U' && ia <= j)
                return false
            end

            pa += one(IA)
        end

        while pb < pbstop
            ib = Browval[pb]

            if (uplo == 'L' && ib >= j) || (uplo == 'U' && ib <= j)
                return false
            end

            pb += one(IB)
        end
    end

    return true
end

function checkpatternsym(P, dX)
    return false
end

function checkpatternsym(P::Union{HermSparseProjectTo{T, I}, SymSparseProjectTo{T, I}}, dX::HermOrSymSparse{T, I}) where {T, I}
    dXP = parent(dX)
    return Symbol(dX.uplo) == P.uplo && checkpatternsym(size(dXP, 2), P.parent.colptr, dXP.colptr, P.parent.rowval, dXP.rowval, dX.uplo)
end

function (P::HermSparseProjectTo{T, I})(dX::HermSparse) where {T, I}
    if checkpatternsym(P, dX)
        return dX
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::SymSparseProjectTo{T, I})(dX::SymSparse) where {T, I}
    if checkpatternsym(P, dX)
        return dX
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::HermSparseProjectTo{T, I})(dX::SymSparse{T, I}) where {T <: Real, I}
    if checkpatternsym(P, dX)
        return Hermitian(parent(dX), P.uplo)
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::SymSparseProjectTo{T, I})(dX::HermSparse{T, I}) where {T <: Real, I}
    if checkpatternsym(P, dX)
        return Symmetric(parent(dX), P.uplo)
    else
        return project!(sparse_from_project(P), dX)
    end
end

#####
##### ChainRules: selupd! for computing sparse gradients
#####

function unwrap(A)
    if A isa Adjoint
        B = parent(A)

        if B isa Transpose
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:C))
        end
    elseif A isa Transpose
        B = parent(A)

        if B isa Adjoint
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:N))
        end
    else
        return (A, Val(:N), Val(:N))
    end
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A Bᴴ + conj(α) B Aᴴ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::HermSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo, A, adjoint(B),      α,  β)
    selupd!(parent(C), C.uplo, B, adjoint(A), conj(α), 1)
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A Bᴴ + α conj(B) Aᵀ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::SymSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo, A,                     adjoint(B),   α, β)
    selupd!(parent(C), C.uplo, adjoint(transpose(B)), transpose(A), α, 1)
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A B + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::SparseMatrixCSC, uplo::Char, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    AP, tA, cA = unwrap(A)
    BP, tB, cB = unwrap(B)
    return selupd_impl!(C, uplo, AP, BP, α, β, tA, cA, tB, cB)
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}) where {tA, cA, tB, cB}
    @assert size(C, 1) == size(C, 2) == length(A) == length(B)

    @inbounds for j in axes(C, 2)
        Bj = cB === :C ? conj(B[j]) : B[j]

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                Ai = cA === :C ? conj(A[i]) : A[i]

                if iszero(β)
                    nonzeros(C)[p] = α * Ai * Bj
                else
                    nonzeros(C)[p] = β * nonzeros(C)[p] + α * Ai * Bj
                end
            end
        end
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}) where {TA, CA, TB, CB}
    @assert size(C, 1) == size(C, 2)

    if TA === :N && TB === :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 2) == size(B, 1)
    elseif TA === :N && TB !== :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 2) == size(B, 2)
    elseif TA !== :N && TB === :N
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 1) == size(B, 1)
    else
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 1) == size(B, 2)
    end

    if TA === :N
        rng = axes(A, 2)
    else
        rng = axes(A, 1)
    end

    if iszero(β)
        fill!(nonzeros(C), β)
    else
        rmul!(nonzeros(C), β)
    end

    for k in rng
        if TA === :N
            Ak = view(A, :, k)
        else
            Ak = view(A, k, :)
        end

        if TB === :N
            Bk = view(B, k, :)
        else
            Bk = view(B, :, k)
        end

        selupd_impl!(C, uplo, Ak, Bk, α, 1, tA, cA, tB, cB)
    end

    return C
end

#####
##### ChainRules: rrule/frule implementations
#####

function mul_rrule_impl(A::HermOrSymSparse, B::DenseVecOrMat, ΔC)
    ΔB = A * ΔC
    ΔA = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, ΔC, B, 1 / 2, 0)
            ΔA
        end
    end
    return ΔA, ΔB
end

function mul_rrule_impl(A::DenseMat, B::HermSparse, ΔC)
    ΔA = ΔC * B
    ΔB = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔB = similar(B)
            selupd!(ΔB, A', ΔC', 1 / 2, 0)
            ΔB
        end
    end
    return ΔA, ΔB
end

function mul_rrule_impl(A::DenseMat, B::SymSparse, ΔC)
    ΔA = ΔC * B
    ΔB = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔB = similar(B)
            selupd!(ΔB, transpose(ΔC), transpose(A), 1 / 2, 0)
            ΔB
        end
    end
    return ΔA, ΔB
end

function dot_rrule_impl(x::StridedVector, A::HermOrSymSparse, y::StridedVector, Ax::StridedVector, Ay::StridedVector, Δz)
    Δx = @thunk Δz * Ay
    Δy = @thunk Δz * Ax

    ΔA = if Δz isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, x, y, Δz / 2, 0)
            ΔA
        end
    end

    return Δx, ΔA, Δy
end

function mul_rrule(A::HermOrSymSparse, B::DenseVecOrMat)
    C = A * B

    function pullback(ΔC)
        ΔA, ΔB = mul_rrule_impl(A, B, ΔC)
        return NoTangent(), ΔA, ΔB
    end

    return C, pullback ∘ unthunk
end

function mul_rrule(A::DenseMat, B::HermOrSymSparse)
    C = A * B

    function pullback(ΔC)
        ΔA, ΔB = mul_rrule_impl(A, B, ΔC)
        return NoTangent(), ΔA, ΔB
    end

    return C, pullback ∘ unthunk
end

function dot_rrule(x::StridedVector, A::HermOrSymSparse, y::StridedVector)
    Ax = A * x
    Ay = A * y
    z = dot(x, Ay)

    function pullback(Δz)
        Δx, ΔA, Δy = dot_rrule_impl(x, A, y, Ax, Ay, Δz)
        return NoTangent(), Δx, ΔA, Δy
    end

    return z, pullback ∘ unthunk
end

function mul_frule_impl(A, B, dA, dB)
    return A * B, dA * B + A * dB
end

function dot_frule_impl(x::StridedVector, A::HermOrSymSparse, y::StridedVector, dx, dA, dy)
    return dot(x, A, y), dot(dx, A, y) + dot(x, A, dy) + dot(x, dA, y)
end

#####
##### ChainRules: frule / rrule dispatches
#####

for T in (HermSparse, SymSparse)
    # A * X
    @eval function ChainRulesCore.frule((_, dA, dX)::Tuple, ::typeof(*), A::$T, X::DenseVecOrMat)
        return mul_frule_impl(A, X, dA, dX)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), A::$T, X::DenseVecOrMat)
        return mul_rrule(A, X)
    end

    # X * A
    @eval function ChainRulesCore.frule((_, dX, dA)::Tuple, ::typeof(*), X::DenseMat, A::$T)
        return mul_frule_impl(X, A, dX, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), X::DenseMat, A::$T)
        return mul_rrule(X, A)
    end

    # dot(x, A, y) - vectors only, matching upstream ChainRules
    @eval function ChainRulesCore.frule((_, dx, dA, dy)::Tuple, ::typeof(dot), x::StridedVector, A::$T, y::StridedVector)
        return dot_frule_impl(x, A, y, dx, dA, dy)
    end

    @eval function ChainRulesCore.rrule(::typeof(dot), x::StridedVector, A::$T, y::StridedVector)
        return dot_rrule(x, A, y)
    end
end
