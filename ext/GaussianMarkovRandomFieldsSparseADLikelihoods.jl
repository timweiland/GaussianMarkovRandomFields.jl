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

# Residual-curvature term C = Σ_k (W r)_k ∇²f_k(x*) = ∇²[Σ_k (W r)_k f_k(x)] at x*.
# `W r` is held constant (evaluated at x*), so this is a plain scalar Hessian with no
# inner Jacobian — computed once, at the primal mode, via sparse forward-mode AD.
function GaussianMarkovRandomFields.residual_curvature(
        lik::GaussianMarkovRandomFields.NonlinearLeastSquaresLikelihood, x_star::AbstractVector
    )
    f = GaussianMarkovRandomFields._residual_function(lik)
    Wr = lik.inv_σ² .* (lik.y .- f(x_star))
    g = x -> sum(Wr .* f(x))
    backend = DI.AutoSparse(
        DI.AutoForwardDiff();
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    return sparse(DI.hessian(g, backend, x_star))
end

end # module
