module GaussianMarkovRandomFieldsSparseADLikelihoods

using GaussianMarkovRandomFields
import DifferentiationInterface as DI
using SparseConnectivityTracer
using SparseMatrixColorings

function GaussianMarkovRandomFields.default_hessian_backend(grad_backend::DI.AbstractADType)
    # Create sparse backend for Hessian computation
    return DI.AutoSparse(
        grad_backend;
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
end

# Also provide a sparse Jacobian backend for models that need J(x)
function GaussianMarkovRandomFields.default_sparse_jacobian_backend()
    # Reuse the package's default gradient backend preference order
    base = GaussianMarkovRandomFields.default_grad_backend()
    return DI.AutoSparse(
        base;
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
end

end # module
