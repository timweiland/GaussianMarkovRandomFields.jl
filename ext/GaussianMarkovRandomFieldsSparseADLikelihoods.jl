module GaussianMarkovRandomFieldsSparseADLikelihoods

using GaussianMarkovRandomFields
import DifferentiationInterface as DI
using SparseConnectivityTracer
using SparseMatrixColorings
using SparseArrays

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

# Sparse Jacobian backend with a fixed (pre-detected) pattern. Detection runs once,
# here, on the primal residual `f` — so it never traces AD-tagged hyperparameters
# (which would hit a Dual×Tracer ambiguity) — and the AutoForwardDiff inner backend
# nests cleanly under an outer ForwardDiff pass, enabling hyperparameter gradients.
function GaussianMarkovRandomFields.known_pattern_jacobian_backend(f, x_probe::AbstractVector)
    pattern = sparse(DI.jacobian_sparsity(f, x_probe, TracerSparsityDetector()))
    return DI.AutoSparse(
        DI.AutoForwardDiff();
        sparsity_detector = DI.ADTypes.KnownJacobianSparsityDetector(pattern),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
end

end # module
