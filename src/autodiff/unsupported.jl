using ChainRulesCore

# Rules whose only job is to stop reverse-mode AD with a useful message.
#
# The ChainRules counterpart of `ext/enzyme/unsupported.jl`, for the same reason:
# left to itself Zygote wanders into the sparse factorization and fails somewhere
# deep in its internals, with an error that names neither the operation nor the
# type the user actually wrote.

"""
    ChainRulesCore.rrule(::typeof(var), d::AbstractGMRF)

Refuse to differentiate marginal variances under ChainRules-based reverse mode.

`var(d) = diag(Q⁻¹)` is differentiable on paper — `∂Σᵢᵢ/∂Q = -(Σ eᵢ eᵢᵀ Σ)` — but
evaluating it needs *full rows* of `Σ` rather than the selected inverse, which
only carries `Q`'s own sparsity pattern. There is no cheap sparse rule to write,
and one built on `selinv` anyway would be silently wrong. Contrast `logdetcov`,
where `∂/∂Q = -Q⁻¹` touches only the pattern and the selected inverse is exact.

This also covers `std`, which is `sqrt.(var(d))`.
"""
function ChainRulesCore.rrule(::typeof(var), d::AbstractGMRF)
    return throw(
        ArgumentError(
            """
            Zygote cannot differentiate `var`/`std` for $(typeof(d).name.name).

            Marginal variances need entries of the covariance outside the \
            precision's sparsity pattern, which selected inversion does not \
            compute. A rule built on it would return a plausible but incorrect \
            gradient rather than an error, so this package refuses instead.

            Use ForwardDiff for marginal-variance gradients. The per-backend \
            support matrix is in the "Automatic Differentiation" reference page \
            of the documentation.
            """
        )
    )
end
