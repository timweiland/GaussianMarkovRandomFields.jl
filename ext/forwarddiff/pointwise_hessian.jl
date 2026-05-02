# Pointwise-diagonal Hessian fast path for `AutoDiffLikelihood`.
#
# When the user provides a `pointwise_loglik_func`, the joint loglik is a sum
# of N per-element terms and the Hessian is diagonal. Computing each diagonal
# entry as a single-variable second derivative via ForwardDiff nests cleanly
# under any outer AD pass (Dual-of-Dual handles fine), so this path
# unblocks nested-AD scenarios where DI.hessian's inner backend either can't
# return Duals (Enzyme/Mooncake) or trips on buffer eltype mismatches under
# Dual-of-Dual.

function GMRFs._pointwise_diagonal_hessian(pointwise_loglik_func, x::AbstractVector)
    # Probe the output eltype: sensitivity may flow through closure-captured
    # Duals (e.g. a hyperparameter being differentiated by an outer AD pass)
    # rather than `x` itself, in which case `eltype(x)` is plain `Float64` but
    # the result is `Dual`. Take the eltype of the function output and
    # promote with `eltype(x)` so we cover both directions of sensitivity.
    T = promote_type(eltype(pointwise_loglik_func(x)), eltype(x))
    diag_vals = Vector{T}(undef, length(x))
    for i in eachindex(x)
        scalar_f = let i = i, x = x
            xi -> pointwise_loglik_func(_replace_index(x, i, xi))[i]
        end
        diag_vals[i] = ForwardDiff.derivative(xj -> ForwardDiff.derivative(scalar_f, xj), x[i])
    end
    return Diagonal(diag_vals)
end

# Replace `x[i]` with `xi`, promoting eltype if needed so the Dual coordinate
# from ForwardDiff doesn't clash with the rest of the vector.
function _replace_index(x::AbstractVector, i::Int, xi)
    T = promote_type(eltype(x), typeof(xi))
    out = T === eltype(x) ? copy(x) : convert(Vector{T}, x)
    out[i] = xi
    return out
end
