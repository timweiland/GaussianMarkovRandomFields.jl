using LinearMaps, SparseArrays

export to_matrix

to_matrix(A::AbstractMatrix) = A
to_matrix(L::LinearMap) = sparse(L)
to_matrix(L::LinearMaps.WrappedMap) = convert(AbstractMatrix, L)
to_matrix(L::LinearMaps.WrappedMap{<:AbstractMatrix}) = L.A_mat
to_matrix(L::LinearMaps.LinearCombination) = mapreduce(to_matrix, +, L.maps)
to_matrix(L::LinearMaps.UniformScalingMap) = spdiagm(0 => fill(L.λ, size(L, 1)))
to_matrix(L::LinearMaps.KroneckerMap) = mapreduce(to_matrix, kron, L.maps)
