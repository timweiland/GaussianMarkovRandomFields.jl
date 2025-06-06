module GaussianMarkovRandomFieldsLDLFactorizations

using GaussianMarkovRandomFields
using LDLFactorizations

using LinearAlgebra, Random

function GaussianMarkovRandomFields.linmap_cholesky_ldl_factorizations(
    A::AbstractMatrix; perm=nothing
    )
    return ldl(A)
end

function GaussianMarkovRandomFields.compute_rand!(
    s::AbstractCholeskySolver{:autodiffable},
    rng::Random.AbstractRNG,
    x::AbstractVector,
    )
    F = s.precision_chol
    randn!(rng, x)
    LDLFactorizations.ldl_dsolve!(F.n, x, sqrt.(F.d))
    LDLFactorizations.ldl_ltsolve!(F.n, x, F.Lp, F.Li, F.Lx)
    permute!(x, F.pinv)
    x .+= GaussianMarkovRandomFields._ensure_dense(
        GaussianMarkovRandomFields.compute_mean(s)
    )
    return x
end

function GaussianMarkovRandomFields.compute_logdetcov(
    s::AbstractCholeskySolver{:autodiffable},
)
    if s.computed_logdetcov !== nothing
        return s.computed_logdetcov
    end
    s.computed_logdetcov = -logdet(s.precision_chol.D)
    return s.computed_logdetcov
end

end
