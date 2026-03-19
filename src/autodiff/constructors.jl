using ChainRulesCore
using SparseArrays
using LinearAlgebra
using LinearMaps

"""
    ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, algorithm)

Automatic differentiation rule for GMRF constructor.

This rrule enables differentiation through GMRF construction, allowing gradients to flow
back to the mean vector and precision matrix parameters. The LinearSolve algorithm is
treated as non-differentiable (NoTangent).

# Arguments
- `μ::AbstractVector`: Mean vector
- `Q::Union{AbstractMatrix, LinearMaps.LinearMap}`: Precision matrix (sparse, dense, or LinearMap)
- `algorithm`: LinearSolve algorithm (treated as non-differentiable)

# Returns
- Forward: Constructed GMRF
- Backward: Pullback function that extracts tangents for μ and Q
"""
function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap}, algorithm)
    # Forward computation - construct GMRF using existing implementation
    x = GMRF(μ, Q, algorithm)

    function GMRF_pullback(x̄)
        # Extract tangents from the GMRF tangent
        μ̄ = x̄.mean
        Q̄ = x̄.precision

        return NoTangent(), μ̄, Q̄, NoTangent()
    end

    return x, GMRF_pullback
end

"""
    ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap})

Automatic differentiation rule for GMRF constructor with default algorithm.

This handles the case where no LinearSolve algorithm is provided (uses default CholeskyFactorization).
"""
function ChainRulesCore.rrule(::Type{GMRF}, μ::AbstractVector, Q::Union{AbstractMatrix, LinearMaps.LinearMap})
    # Forward computation - construct GMRF using existing implementation
    x = GMRF(μ, Q)

    function GMRF_pullback(x̄)
        # Extract tangents from the GMRF tangent
        μ̄ = x̄.mean
        Q̄ = x̄.precision

        return NoTangent(), μ̄, Q̄
    end

    return x, GMRF_pullback
end

# ConstrainedGMRF constructor rrule.
# The constructor precomputes Ã=Q⁻¹A', L_c=chol(AÃ'), constrained_mean, and
# log_constraint_correction — all of which depend on the base_gmrf's μ and Q.
# A and e are treated as non-differentiable (they define model structure, not hyperparameters).
function ChainRulesCore.rrule(
        ::Type{ConstrainedGMRF}, base_gmrf::AbstractGMRF, A::AbstractMatrix, e::AbstractVector
    )
    result = ConstrainedGMRF(base_gmrf, A, e)

    function ConstrainedGMRF_pullback(ȳ)
        # Start with the base_gmrf tangent passed through directly (e.g., from logpdf)
        incoming_base = ȳ.base_gmrf
        if !_is_zero_tangent(incoming_base) && incoming_base isa Tangent
            μ̄_base = incoming_base.mean
            Q̄_base = incoming_base.precision
        else
            μ̄_base = NoTangent()
            Q̄_base = NoTangent()
        end

        # Gradient through constrained_mean: μ_c = μ - Ã'(AÃ')⁻¹(Aμ - e)
        # ∂μ_c/∂μ = P = I - Ã'S⁻¹A, so pullback needs P^T = I - A^TS⁻¹Ã
        # P^T * μ̄_c = μ̄_c - A^T * S⁻¹ * Ã * μ̄_c  (note: P ≠ P^T since Q⁻¹A^T ≠ A)
        if !_is_zero_tangent(ȳ.constrained_mean)
            μ̄_c = collect(ȳ.constrained_mean)
            v = result.L_c \ (result.A_tilde_T' * μ̄_c)  # S⁻¹ * Ã * μ̄_c
            μ̄_proj = μ̄_c - result.constraint_matrix' * v  # P^T * μ̄_c
            μ̄_base = _is_zero_tangent(μ̄_base) ? μ̄_proj : collect(μ̄_base) + μ̄_proj
        end

        # Gradient through log_constraint_correction:
        # c = 0.5*(r*log(2π) + logdet(L_c) + resid'L_c⁻¹*resid) - 0.5*logdet(AA')
        # where resid = e - Aμ, L_c = chol(AÃ'), Ã = Q⁻¹A'
        if !_is_zero_tangent(ȳ.log_constraint_correction)
            c̄ = ȳ.log_constraint_correction
            resid = result.constraint_vector - result.constraint_matrix * mean(result.base_gmrf)

            # ∂c/∂μ = A'(AÃ')⁻¹(Aμ - e) * c̄  (from the quadratic term)
            μ̄_corr = c̄ * (result.constraint_matrix' * (result.L_c \ (-resid)))
            μ̄_base = _is_zero_tangent(μ̄_base) ? μ̄_corr : collect(μ̄_base) + μ̄_corr

            # ∂c/∂Q: through Ã = Q⁻¹A' which appears in L_c = AÃ'
            # Using ∂(Q⁻¹)/∂Q = -Q⁻¹(dQ)Q⁻¹:
            # ∂c/∂Q = -0.5 * c̄ * Ã * ((AÃ')⁻¹ - (AÃ')⁻¹*resid*resid'*(AÃ')⁻¹) * Ã'
            S_inv = inv(result.L_c)  # (AÃ')⁻¹, small m×m matrix
            w = S_inv * resid
            M = c̄ * (-0.5) * result.A_tilde_T * (S_inv - w * w') * result.A_tilde_T'
            Q̄_base = _is_zero_tangent(Q̄_base) ? M : Q̄_base + M
        end

        base_tangent = Tangent{typeof(base_gmrf)}(;
            mean = μ̄_base,
            precision = Q̄_base,
            information_vector = NoTangent(),
            Q_sqrt = NoTangent(),
            linsolve_cache = NoTangent(),
            rbmc_strategy = NoTangent(),
        )

        return NoTangent(), base_tangent, NoTangent(), NoTangent()
    end

    return result, ConstrainedGMRF_pullback
end
