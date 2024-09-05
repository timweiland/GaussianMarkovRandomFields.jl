using LinearMaps, SparseArrays

export to_matrix

to_matrix(A::AbstractMatrix) = A
to_matrix(L::LinearMap) = sparse(L)
to_matrix(L::LinearMaps.WrappedMap) = convert(AbstractMatrix, L)
to_matrix(L::LinearMaps.LinearCombination) = mapreduce(to_matrix, +, L.maps)
