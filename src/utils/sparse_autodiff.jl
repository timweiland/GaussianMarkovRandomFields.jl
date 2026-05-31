"""
    sparse_autodiff.jl

Stubs for sparse autodiff backends. Concrete methods are provided by package
extensions when optional dependencies are loaded (e.g., SparseConnectivityTracer
and SparseMatrixColorings) via DifferentiationInterface.AutoSparse.
"""

function default_sparse_jacobian_backend end

"""
    known_pattern_jacobian_backend(f, x_probe) -> AutoSparse

Detect the Jacobian sparsity pattern of `f` at `x_probe` once and return a sparse
`AutoSparse` backend that reuses that fixed pattern (skipping per-call detection).
The detection runs here on the supplied — necessarily primal — residual, so it never
sees AD-tagged values, and the `AutoForwardDiff` inner backend nests cleanly under an
outer forward-mode AD pass over hyperparameters.

Concrete method provided by the SparseConnectivityTracer + SparseMatrixColorings extension.
"""
function known_pattern_jacobian_backend end
