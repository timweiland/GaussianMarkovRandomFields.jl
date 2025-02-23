using SparseArrays

export TakahashiStrategy, compute_variance

"""
    TakahashiStrategy()

Takahashi recursions [1] for computing the marginal variances of a GMRF.
Highly accurate, but computationally expensive.
Uses `SparseInverseSubset.jl`.
"""
struct TakahashiStrategy <: AbstractVarianceStrategy
    function TakahashiStrategy()
        new()
    end
end

function compute_variance(::TakahashiStrategy, solver::CholeskySolver)
    if solver.precision_chol isa SparseArrays.CHOLMOD.Factor
        return diag(sparseinv(solver.precision_chol, depermute = true)[1])
    else
        return diag(sparseinv(sparse(to_matrix(solver.precision)), depermute = true)[1])
    end
end
