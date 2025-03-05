module GMRFsSparseJacobian

using GMRFs
using LinearMaps, Symbolics, SparseDiffTools
using SparseArrays

"""
    sparse_jacobian_map(
        f!::Function,
        x₀::AbstractVector{T},
        N_outputs::Int;
        colors::AbstractVector{Int64} = nothing,
        jac::SparseMatrixCSC{T} = nothing,
    )

Construct a linear map representing the sparse Jacobian of `f!` at `x₀`.

# Arguments
- `f!::Function`: Function to differentiate. Must be in-place
        with signature `f!(output, input)`.
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
- `N_outputs::Int`: Output dimension of `f!`.

# Keyword arguments
- `colors::AbstractVector{Int64}`: Optional pre-computed coloring for
                                   sparse differentiation.
- `jac::SparseMatrixCSC{T}`: Optional pre-computed sparsity pattern for the
                             Jacobian.
"""
function GMRFs.sparse_jacobian_map(
    f!::Function,
    x₀::AbstractVector{T},
    N_outputs::Int;
    colors::AbstractVector{Int64} = nothing,
    jac::SparseMatrixCSC{T} = nothing,
    ) where {T}
    N_outputs > 0 || throw(ArgumentError("N_outputs must be positive"))

    output = rand(N_outputs)

    if jac === nothing
        sparsity_pattern = Symbolics.jacobian_sparsity(f!, output, x₀)
        jac = Float64.(sparsity_pattern)
    end
    if colors === nothing
        colors = matrix_colors(jac)
    end
    forwarddiff_color_jacobian!(jac, f!, x₀, colorvec = colors)

    return LinearMaps.LinearMap(jac), colors
end

end
