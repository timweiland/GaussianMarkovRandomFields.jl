"""
    sparse_autodiff.jl

Stubs for sparse autodiff backends. Concrete methods are provided by package
extensions when optional dependencies are loaded (e.g., SparseConnectivityTracer
and SparseMatrixColorings) via DifferentiationInterface.AutoSparse.
"""

function default_sparse_jacobian_backend end
