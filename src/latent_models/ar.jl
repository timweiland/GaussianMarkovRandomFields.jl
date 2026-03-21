using SparseArrays
using LinearAlgebra
using LinearSolve

export ARModel, AR1Model

_default_ar_alg(::Val{1}) = LDLtFactorization()
_default_ar_alg(::Val) = CHOLMODFactorization()

"""
    _durbin_levinson(pacf)

Convert partial autocorrelation (PACF) values to AR coefficients via the Durbin-Levinson recursion.

Returns `(phi, phi_history)` where:
- `phi`: final AR(P) coefficient vector
- `phi_history`: vector of vectors, `phi_history[k]` = AR(k) coefficients at step k
  (needed for boundary rows of the precision matrix)
"""
function _durbin_levinson(pacf)
    P = length(pacf)
    T = eltype(pacf)

    phi = zeros(T, P)
    phi_history = Vector{Vector{T}}(undef, P)

    phi[1] = pacf[1]
    phi_history[1] = [pacf[1]]

    for k in 2:P
        phi_prev = copy(phi)
        phi[k] = pacf[k]
        for j in 1:(k - 1)
            phi[j] = phi_prev[j] - pacf[k] * phi_prev[k - j]
        end
        phi_history[k] = phi[1:k]
    end

    return (phi, phi_history)
end

"""
    ARModel{P}(n::Int; alg=<auto>, constraint=nothing)

A stationary autoregressive model of order P using the PACF parameterization.

The AR(P) model represents a temporal process: x[t] = φ₁x[t-1] + ... + φₚx[t-P] + ε[t].
Instead of parameterizing directly by the AR coefficients φ, the model uses partial
autocorrelation (PACF) values θ₁, ..., θₚ which guarantee stationarity when |θₖ| < 1.
The k-th PACF value θₖ is the correlation between x[t] and x[t-k] after removing the
linear effect of the intervening lags x[t-1], ..., x[t-k+1]. The PACF values are converted
to AR coefficients internally via the Durbin-Levinson recursion.

# Type Aliases
- `AR1Model = ARModel{1}` — first-order autoregressive (backward compatible)

# Hyperparameters
- **P=1**: `τ` (precision, τ > 0) and `ρ` (correlation, |ρ| < 1)
- **P≥2**: `τ` (precision) and `pacf1, pacf2, ..., pacfP` (PACF values, each in (-1, 1))

# Precision Matrix
Constructed as `Q = τ · L' · D · L` where L is unit lower triangular with bandwidth P
and D is diagonal, encoding the stationary AR(P) structure.

- **P=1**: Returns `SymTridiagonal` (efficient tridiagonal solver)
- **P≥2**: Returns `SparseMatrixCSC` (banded with bandwidth P)

# Example
```julia
# AR(1) — backward compatible
model = AR1Model(100)
gmrf = model(τ=2.0, ρ=0.8)

# AR(2) via PACF
model2 = ARModel{2}(100)
gmrf2 = model2(τ=1.0, pacf1=0.7, pacf2=-0.3)

# AR(3)
model3 = ARModel{3}(100)
gmrf3 = model3(τ=1.0, pacf1=0.5, pacf2=-0.3, pacf3=0.2)
```
"""
struct ARModel{P, Alg, C, L} <: LatentModel
    n::Int
    alg::Alg
    constraint::C
    levels::L

    function ARModel{P, Alg, C, L}(n::Int, alg::Alg, constraint::C, levels::L) where {P, Alg, C, L}
        P isa Int && P >= 1 || throw(ArgumentError("AR order P must be a positive integer, got P=$P"))
        n > 0 || throw(ArgumentError("Length n must be positive, got n=$n"))
        if P >= 2
            n > P || throw(ArgumentError("AR$P requires length n > $P, got n=$n"))
        end
        return new{P, Alg, C, L}(n, alg, constraint, levels)
    end
end

function ARModel{P}(n::Int; alg = _default_ar_alg(Val(P)), constraint = nothing, levels = nothing) where {P}
    processed_constraint = _process_constraint(constraint, n)
    return ARModel{P, typeof(alg), typeof(processed_constraint), typeof(levels)}(n, alg, processed_constraint, levels)
end

"""Backward-compatible alias for `ARModel{1}`."""
const AR1Model = ARModel{1}

function Base.length(model::ARModel)
    return model.n
end

function hyperparameters(model::ARModel{1})
    return (τ = Real, ρ = Real)
end

function hyperparameters(::ARModel{P}) where {P}
    names = ntuple(i -> i == 1 ? :τ : Symbol("pacf$(i - 1)"), P + 1)
    return NamedTuple{names}(ntuple(_ -> Real, P + 1))
end

function _validate_ar1_parameters(; τ::Real, ρ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    abs(ρ) < 1 || throw(ArgumentError("Correlation parameter ρ must satisfy |ρ| < 1, got ρ=$ρ"))
    return nothing
end

function _validate_ar_parameters(; τ::Real, pacf_values)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    for (k, θ) in enumerate(pacf_values)
        abs(θ) < 1 || throw(ArgumentError("PACF parameter pacf$k must satisfy |pacf$k| < 1, got pacf$k=$θ"))
    end
    return nothing
end

# Specialized precision matrix for P=1: returns SymTridiagonal (unchanged from original AR1)
function precision_matrix(model::ARModel{1}; τ::Real, ρ::Real, kwargs...)
    _validate_ar1_parameters(; τ = τ, ρ = ρ)

    n = model.n
    T = promote_type(typeof(τ), typeof(ρ))

    main_diag = map(1:n) do i
        (i == 1 || i == n) ? T(τ) : T((1 + ρ^2) * τ)
    end

    off_diag = fill(-T(ρ * τ), n - 1)

    return SymTridiagonal(main_diag, off_diag)
end

# Generic precision matrix for P >= 2: returns SparseMatrixCSC
# Uses Q = τ * L' * D * L where:
#   L = unit lower triangular (bandwidth P) encoding AR structure
#   D = diagonal encoding conditional variances with stationary initialization
function precision_matrix(model::ARModel{P}; τ::Real, kwargs...) where {P}
    pacf_values = ntuple(k -> kwargs[Symbol("pacf$k")]::Real, Val(P))
    _validate_ar_parameters(; τ = τ, pacf_values = pacf_values)

    n = model.n
    T = promote_type(typeof(τ), map(typeof, pacf_values)...)
    T = promote_type(T, Float64)

    # PACF → AR coefficients (with history for boundary rows)
    phi, phi_history = _durbin_levinson(collect(T, pacf_values))

    # Build D diagonal: D[t] = prod(1 - θ_k² for k = min(t-1,P)+1 : P)
    # D[1] = prod(1 - θ_k² for k=1:P)  (full product = σ_e²)
    # D[t] for 1 < t ≤ P: prod(1 - θ_k² for k=t:P)
    # D[t] for t > P: 1.0
    D_diag = map(1:n) do t
        if t == 1
            prod(1 - T(θ)^2 for θ in pacf_values)
        elseif t <= P
            prod(1 - T(pacf_values[k])^2 for k in t:P)
        else
            T(1)
        end
    end

    # Build L as sparse unit lower triangular (bandwidth P)
    # L[t, t] = 1 for all t
    # For t ≤ P: L[t, t-k] = -φ_k^(t-1) using AR(t-1) coefficients
    # For t > P: L[t, t-k] = -φ_k using final AR(P) coefficients
    I_idx = Int[]
    J_idx = Int[]
    V_vals = T[]

    # Diagonal entries
    for t in 1:n
        push!(I_idx, t)
        push!(J_idx, t)
        push!(V_vals, T(1))
    end

    # Sub-diagonal entries
    for t in 2:n
        if t <= P
            # Use AR(t-1) coefficients from phi_history
            ar_coeffs = phi_history[t - 1]
            for k in 1:(t - 1)
                push!(I_idx, t)
                push!(J_idx, t - k)
                push!(V_vals, -ar_coeffs[k])
            end
        else
            # Use final AR(P) coefficients
            for k in 1:P
                push!(I_idx, t)
                push!(J_idx, t - k)
                push!(V_vals, -phi[k])
            end
        end
    end

    L = sparse(I_idx, J_idx, V_vals, n, n)
    D = spdiagm(D_diag)

    # Q = τ * L' * D * L
    Q = T(τ) * (L' * D * L)

    return Q
end

function mean(model::ARModel; kwargs...)
    return zeros(model.n)
end

function constraints(model::ARModel; kwargs...)
    return model.constraint
end

function model_name(::ARModel{1})
    return :ar1
end

function model_name(::ARModel{P}) where {P}
    return Symbol("ar$P")
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
