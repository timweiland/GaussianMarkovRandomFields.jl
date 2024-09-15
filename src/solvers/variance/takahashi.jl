export TakahashiStrategy, compute_variance

struct TakahashiStrategy <: AbstractVarianceStrategy
    function TakahashiStrategy()
        new()
    end
end

function compute_variance(::TakahashiStrategy, solver::CholeskySolver)
    return diag(sparseinv(solver.precision_chol, depermute = true)[1])
end
