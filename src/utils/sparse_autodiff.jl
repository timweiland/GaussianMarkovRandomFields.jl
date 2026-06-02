"""
    sparse_autodiff.jl

Stubs for sparse autodiff backends. Concrete methods are provided by package
extensions when optional dependencies are loaded (e.g., SparseConnectivityTracer
and SparseMatrixColorings) via DifferentiationInterface.AutoSparse.

The fallbacks below use `Vararg{Any}`/`Any` so that any extension-provided method
with a concrete signature takes precedence by Julia dispatch — no method
overwriting occurs. The fallbacks exist so callers get a clear error (and JET
can see a method) when the relevant extension is not loaded.
"""

function default_sparse_jacobian_backend(args::Vararg{Any}; kwargs...)
    throw(
        ArgumentError(
            "Sparse Jacobian backend not available. " *
                "Load SparseConnectivityTracer and SparseMatrixColorings " *
                "to activate the AutoSparse backend."
        )
    )
end

"""
    known_pattern_jacobian_backend(f, x_probe) -> AutoSparse

Detect the Jacobian sparsity pattern of `f` at `x_probe` once and return a sparse
`AutoSparse` backend that reuses that fixed pattern (skipping per-call detection).
The detection runs here on the supplied — necessarily primal — residual, so it never
sees AD-tagged values, and the `AutoForwardDiff` inner backend nests cleanly under an
outer forward-mode AD pass over hyperparameters.

Concrete method provided by the SparseConnectivityTracer + SparseMatrixColorings extension.
"""
function known_pattern_jacobian_backend(args::Vararg{Any}; kwargs...)
    throw(
        ArgumentError(
            "Sparse Jacobian backend not available. " *
                "Load SparseConnectivityTracer and SparseMatrixColorings " *
                "to activate the AutoSparse backend."
        )
    )
end

"""
    residual_curvature(lik::NonlinearLeastSquaresLikelihood, x_star) -> SparseMatrixCSC

The residual-curvature term `C = Σ_k (W r)_k ∇²f_k(x*)` of a Gauss–Newton least-squares
likelihood, evaluated at the primal mode `x*` (`W = Diagonal(inv_σ²)`, `r = y - f(x*)`).

This is the difference between the true loglik Hessian and the Gauss–Newton Hessian
`-JᵀWJ`. It is the Hessian of the scalar `x -> Σ_k (W r)_k f_k(x)` (with `W r` held
fixed), so it carries no inner Jacobian and is computed as an ordinary sparse Hessian.
The implicit-function mode sensitivity must be solved with the true Hessian
`Q_prior + JᵀWJ - C`, so subtracting `C` from the Gauss–Newton posterior precision makes
hyperparameter gradients exact even for residuals nonlinear in the latent field.

Concrete method provided by the SparseConnectivityTracer + SparseMatrixColorings extension.
"""
function residual_curvature(args::Vararg{Any}; kwargs...)
    throw(
        ArgumentError(
            "Sparse Jacobian backend not available. " *
                "Load SparseConnectivityTracer and SparseMatrixColorings " *
                "to activate the AutoSparse backend."
        )
    )
end
