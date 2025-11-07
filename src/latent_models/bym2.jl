using SparseArrays
using LinearAlgebra
using LinearSolve

export BYM2Model

"""
    BYM2Model(adjacency::AbstractMatrix; regularization::Float64 = 1e-5, normalize_var::Bool = true, singleton_policy::Symbol = :gaussian, alg=CHOLMODFactorization(), additional_constraints=nothing)

Besag-York-Mollié model with improved BYM2 parameterization (Riebler et al. 2016).

The BYM2 model is a reparameterization of the classic BYM model for disease mapping that
improves interpretability and facilitates prior specification. It combines spatial (ICAR)
and unstructured (IID) random effects with intuitive mixing and precision parameters.

# Mathematical Description

The BYM2 model is a 2n-dimensional latent field stacking:
1. A scaled spatial component: u* ~ ICAR with variance-normalized precision Q*
2. An unstructured component: v* ~ N(0, I)

The block-diagonal precision matrix is:
```
Q = [ τ/(1-φ) * Q*    0           ]
    [ 0               τ/φ * I     ]
```

In the linear predictor, effects are added: η[i] = ... + u*[i] + v*[i]

This parameterization ensures:
- Var(u*[i]) ≈ (1-φ)/τ (spatial variance)
- Var(v*[i]) = φ/τ (unstructured variance)
- Var(u*[i] + v*[i]) = 1/τ (total variance)
- φ controls the proportion of variance that is unstructured
- φ = 0: pure spatial model (like scaled Besag)
- φ = 1: pure unstructured model (like scaled IID)

# Hyperparameters
- `τ`: Overall precision parameter (τ > 0)
- `φ`: Mixing parameter (0 < φ < 1), proportion of unstructured variance

# Fields
- `besag::BesagModel`: The spatial component (variance-normalized)
- `iid::IIDModel`: The unstructured component
- `n::Int`: Number of spatial units
- `alg::Alg`: LinearSolve algorithm for solving linear systems

# Example
```julia
# 4-node cycle graph
W = sparse(Bool[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0])
model = BYM2Model(W)

# Returns 2n-dimensional ConstrainedGMRF with sum-to-zero constraint on spatial component
# Components 1:n are spatial (u*), components (n+1):2n are unstructured (v*)
gmrf = model(τ=1.0, φ=0.5)  # Equal spatial and unstructured variance

# More spatial smoothing (90% spatial, 10% unstructured)
gmrf = model(τ=1.0, φ=0.1)

# More unstructured variation (10% spatial, 90% unstructured)
gmrf = model(τ=1.0, φ=0.9)
```

# References
Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016).
An intuitive Bayesian spatial model for disease mapping that accounts for scaling.
Statistical Methods in Medical Research, 25(4), 1145-1165.
"""
struct BYM2Model{Alg} <: LatentModel
    besag::BesagModel
    iid::IIDModel
    n::Int
    alg::Alg

    function BYM2Model(
            adjacency::AbstractMatrix;
            regularization::Float64 = 1.0e-5,
            normalize_var = Val{true}(),
            singleton_policy = Val{:gaussian}(),
            alg = CHOLMODFactorization(),
            additional_constraints = nothing,
        )
        # Validate normalize_var - BYM2 requires variance normalization
        if normalize_var !== Val{true}()
            throw(ArgumentError("BYM2Model requires variance normalization (normalize_var must be Val{true}())"))
        end

        # Create variance-normalized Besag component
        besag = BesagModel(
            adjacency;
            regularization = regularization,
            normalize_var = normalize_var,
            singleton_policy = singleton_policy,
            alg = alg,
            additional_constraints = additional_constraints,
        )

        n = length(besag)

        # Create IID component (no constraint - Besag handles sum-to-zero)
        iid = IIDModel(n; alg = alg, constraint = nothing)

        return new{typeof(alg)}(besag, iid, n, alg)
    end
end

function Base.length(model::BYM2Model)
    return 2 * model.n  # Stacked: [u* (spatial); v* (unstructured)]
end

function hyperparameters(model::BYM2Model)
    return (τ = Real, φ = Real)
end

function _validate_bym2_parameters(; τ::Real, φ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    0 < φ < 1 || throw(ArgumentError("Mixing parameter φ must be in (0, 1), got φ=$φ"))

    # Warn about extreme values that might cause numerical issues
    if φ < 1e-6
        @warn "φ is very close to 0 (φ=$φ), which may cause numerical instability. Consider using a pure Besag model instead."
    elseif φ > 1 - 1e-6
        @warn "φ is very close to 1 (φ=$φ), which may cause numerical instability. Consider using a pure IID model instead."
    end

    return nothing
end

function precision_matrix(model::BYM2Model; τ::Real, φ::Real, kwargs...)
    _validate_bym2_parameters(; τ = τ, φ = φ)

    # Get the scaled Besag precision (Q* already normalized by BesagModel)
    # We need to pass τ=1.0 to get Q* and then scale it ourselves
    Q_besag_scaled = precision_matrix(model.besag; τ = 1.0)

    # Get IID precision (diagonal)
    Q_iid = precision_matrix(model.iid; τ = 1.0)

    # Apply BYM2 scaling
    # Top block: τ/(1-φ) * Q*
    # Bottom block: τ/φ * I
    Q_spatial = (τ / (1 - φ)) * Q_besag_scaled
    Q_unstructured = (τ / φ) * Q_iid

    # Build block-diagonal precision matrix
    return _blockdiag(Q_spatial, Q_unstructured)
end

function mean(model::BYM2Model; kwargs...)
    return zeros(2 * model.n)
end

function constraints(model::BYM2Model; kwargs...)
    # Get constraints from Besag component
    constraint_info = constraints(model.besag; kwargs...)

    if constraint_info === nothing
        return nothing
    end

    A_besag, e_besag = constraint_info
    n_constraints = size(A_besag, 1)

    # Expand constraints to 2n-dimensional space
    # Constraints only apply to spatial component (first n elements)
    A_expanded = zeros(n_constraints, 2 * model.n)
    A_expanded[:, 1:model.n] = A_besag

    return (A_expanded, e_besag)
end

function model_name(::BYM2Model)
    return :bym2
end

# Helper function for block diagonal construction (reuse from combined.jl pattern)
function _blockdiag(M1::AbstractMatrix, M2::AbstractMatrix)
    n1, m1 = size(M1)
    n2, m2 = size(M2)

    # Convert to sparse
    M1_sparse = M1 isa SparseMatrixCSC ? M1 : sparse(M1)
    M2_sparse = M2 isa SparseMatrixCSC ? M2 : sparse(M2)

    # Extract nonzeros
    I1, J1, V1 = findnz(M1_sparse)
    I2, J2, V2 = findnz(M2_sparse)

    # Offset second block
    I2_offset = I2 .+ n1
    J2_offset = J2 .+ m1

    # Combine
    I_combined = vcat(I1, I2_offset)
    J_combined = vcat(J1, J2_offset)
    V_combined = vcat(V1, V2)

    return sparse(I_combined, J_combined, V_combined, n1 + n2, m1 + m2)
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
