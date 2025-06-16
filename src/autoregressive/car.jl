using LinearAlgebra, SparseArrays

export generate_car_model

@doc raw"""
    generate_car_model(W::SparseMatrixCSC, ρ::Real; σ=1.0, μ=nothing)

Generate a conditional autoregressive model (CAR) in GMRF form from an adjacency
matrix.

### Input

- `W` -- Adjacency / weight matrix. Specifies the conditional dependencies
         between variables
- `ρ` -- Weighting factor of the inter-node dependencies. Fulfills 0 < ρ < 1.
- `σ` -- Variance scaling factor (i.e. output scale)
- `μ` -- Mean vector

### Output

A `GMRF` with the corresponding mean and precision.

### Algorithm

The CAR is constructed using a variant of the graph Laplacian, i.e.

```math
Q = \sigma^{-1} \cdot (W 1 - ρ \cdot W).
```
"""
function generate_car_model(W::SparseMatrixCSC, ρ::Real; σ = 1.0, μ = nothing, solver_blueprint=DefaultSolverBlueprint())
    if (ρ >= 1) || (ρ < 0)
        throw(ArgumentError("Expected 0 < ρ < 1."))
    end
    N = size(W, 2)
    D = Diagonal(W * ones(N))
    Q = (D - ρ * W) ./ σ
    if μ === nothing
        μ = spzeros(N)
    end
    return GMRF(μ, Q, solver_blueprint)
end
