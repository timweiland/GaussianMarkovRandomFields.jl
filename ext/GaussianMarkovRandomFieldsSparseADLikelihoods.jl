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

# Hessian analogue of `known_pattern_jacobian_backend`. `g` must be scalar-valued.
# Detection runs once here on the primal residual sum, so the pattern (which depends only
# on the residual structure, not on σ/θ/x*) is reused across materializations.
function GaussianMarkovRandomFields.known_pattern_hessian_backend(g, x_probe::AbstractVector)
    pattern = sparse(DI.hessian_sparsity(g, x_probe, TracerSparsityDetector()))
    return DI.AutoSparse(
        DI.AutoForwardDiff();
        sparsity_detector = DI.ADTypes.KnownHessianSparsityDetector(pattern),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
end

# Build the backend from an already-known *structural* pattern (no detection). Used by the
# IFT θ-gradient path to reuse a prior's own Hessian sparsity — which works even when the
# density can't be traced (e.g. it solves an ODE), since detection never runs here.
function GaussianMarkovRandomFields.known_pattern_hessian_backend(pattern::SparseMatrixCSC)
    P = SparseMatrixCSC(pattern.m, pattern.n, copy(pattern.colptr), copy(pattern.rowval), ones(Bool, nnz(pattern)))
    return DI.AutoSparse(
        DI.AutoForwardDiff();
        sparsity_detector = DI.ADTypes.KnownHessianSparsityDetector(P),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
end

# Residual-curvature term C = Σ_k (W r)_k ∇²f_k(x*) = ∇²[Σ_k (W r)_k f_k(x)] at x*.
# `W r` is held constant (evaluated at x*), so this is a plain scalar Hessian with no
# inner Jacobian. Its sparsity pattern is θ- and x*-independent, so it is detected once
# and cached on the model (`lik.hess_backend`); only the numeric Hessian is recomputed here.
function GaussianMarkovRandomFields.residual_curvature(
        lik::GaussianMarkovRandomFields.NonlinearLeastSquaresLikelihood, x_star::AbstractVector
    )
    f = GaussianMarkovRandomFields._residual_function(lik)
    Wr = lik.inv_σ² .* (lik.y .- f(x_star))
    g = x -> sum(Wr .* f(x))
    return sparse(DI.hessian(g, lik.hess_backend, x_star))
end

# Structural sparsity of a `StructuredLatentPrior` factor group's K×K Hessian, traced once per group
# at construction so the per-factor scatter only touches the real coupling (a diagonal-covariance
# block factor stays sparse instead of needing a dense block). Local detection handles the
# value-dependent branches inside distribution constructors (e.g. `check_args`).
function GaussianMarkovRandomFields._factor_group_sparsity(
        grp::GaussianMarkovRandomFields.LatentFactorGroup{K}, θ, ::DI.AbstractADType
    ) where {K}
    try
        x_probe = Float64[0.5 + i for i in 1:K]
        H = DI.hessian_sparsity(v -> grp.logp(v, θ), x_probe, TracerLocalSparsityDetector())
        return collect(Bool, H .!= 0)
    catch
        # A factor whose log-density can't be traced falls back to a dense block (then the pattern
        # must contain the full K×K block, as before this detection existed).
        return fill(true, K, K)
    end
end

end # module
