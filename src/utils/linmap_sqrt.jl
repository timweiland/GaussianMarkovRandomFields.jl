using LinearMaps, LinearAlgebra

export linmap_sqrt

function linmap_sqrt(A::LinearMap)
    if issymmetric(A) && isposdef(A)
        return LinearMap(cholesky(to_matrix(A)))
    end
    throw(ArgumentError("No square root available for $A"))
end
linmap_sqrt(L::LinearMaps.LinearCombination) = hcat([linmap_sqrt(Li) for Li in L.maps]...)
