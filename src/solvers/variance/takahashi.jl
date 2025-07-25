using SparseArrays
using SelectedInversion

export TakahashiStrategy, compute_variance

"""
    TakahashiStrategy()

Takahashi recursions [1] for computing the marginal variances of a GMRF.
Highly accurate, but computationally expensive.
Uses `SelectedInversion.jl`.
"""
struct TakahashiStrategy <: AbstractVarianceStrategy
    function TakahashiStrategy()
        new()
    end
end

function compute_variance(::TakahashiStrategy, solver::AbstractCholeskySolver)
    if solver.precision_chol isa SparseArrays.CHOLMOD.Factor
        return selinv_diag(solver.precision_chol)
    else
        return diag(inv(Array(to_matrix(solver.precision))))
    end
end
