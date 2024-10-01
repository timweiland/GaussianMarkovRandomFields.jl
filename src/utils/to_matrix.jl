using LinearMaps, SparseArrays, Memoize

export to_matrix

to_matrix(A::AbstractMatrix) = A
@memoize to_matrix(L::LinearMap) = sparse(L)
to_matrix(L::LinearMaps.WrappedMap) = convert(AbstractMatrix, L)
@memoize to_matrix(L::LinearMaps.LinearCombination) = mapreduce(to_matrix, +, L.maps)
to_matrix(L::LinearMaps.UniformScalingMap) = spdiagm(0 => fill(L.λ, size(L, 1)))
