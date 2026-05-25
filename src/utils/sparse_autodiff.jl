"""
    sparse_autodiff.jl

Stubs for sparse autodiff backends. Concrete methods are provided by package
extensions when optional dependencies are loaded (e.g., SparseConnectivityTracer
and SparseMatrixColorings) via DifferentiationInterface.AutoSparse.

The fallback below uses `Vararg{Any}`/`Any` so that any extension-provided method
with a concrete signature takes precedence by Julia dispatch — no method
overwriting occurs. The fallback exists so callers get a clear error (and JET
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
