# Automatic Differentiation Reference

GaussianMarkovRandomFields.jl supports gradient-based inference and optimization
through several AD backends. Which backend to reach for depends on **both** the
operation and the GMRF type — support is not uniform, and the differences are
large enough to matter in practice.

## Choosing a backend

**Start with ForwardDiff.jl.** It is the only backend that currently handles every
GMRF type and every differentiable operation in this package, and it is genuinely
fast for hyperparameter optimization, where the input is a handful of
hyperparameters and forward mode is the right shape for the problem.

Reach for a reverse-mode backend when the parameter vector is large enough that
forward mode's per-parameter cost dominates:

- **Zygote.jl** for `GMRF` and `WorkspaceGMRF` priors.
- **Mooncake.jl** for `ChordalGMRF`, which is what the chordal backend is built
  for.
- **Enzyme.jl** for `logpdf` and `logdetcov` when its constraints suit you. It is
  the fastest of the three where it applies, but it applies to the least.

## Support matrix

Checked against finite differences with a precision matrix whose Cholesky factor
has genuine fill-in — the Enzyme column on Julia 1.10.10, 1.11.9 and 1.12.6, the
other columns on 1.12.6. The fill-in matters: a tridiagonal AR(1) precision has
none, and several of the failures below are invisible with one.

Mooncake is not in the table; it is covered by the package's own
`ChordalGMRF` test suite rather than by this audit.

| Operation | ForwardDiff | Zygote | Enzyme |
|---|---|---|---|
| `logdetcov(::GMRF)` | ✅ | ❌ raises | ✅ |
| `logpdf(::GMRF, z)` | ✅ | ✅ | ✅ |
| `gaussian_approximation` (`GMRF`) | ✅ | ✅ | ❌ raises |
| `logdetcov(::ChordalGMRF)` | ✅ | ❌ raises | ✅ except Julia 1.11 |
| `logpdf(::ChordalGMRF, z)` | ✅ | ⚠️ **silently wrong** | ✅ except Julia 1.11 |
| `gaussian_approximation` (`ChordalGMRF`) | ✅ | ⚠️ **silently wrong** | ❌ raises |
| `logdetcov(::WorkspaceGMRF)` | ✅ | ❌ raises | ✅ |
| `logpdf(::WorkspaceGMRF, z)` | ✅ | ✅ | ✅ |
| `gaussian_approximation` (`WorkspaceGMRF`) | ✅ | ✅ | ❌ raises |
| `var` / `std` (any type) | ❌ raises | ❌ raises | ❌ raises |

✅ matches finite differences · ❌ raises an error · ⚠️ returns a plausible but
incorrect number

Every ❌ in this table is an explicit, actionable error. Combinations that cannot
be supported raise rather than falling back to differentiating a sparse
factorization, because doing the latter produces wrong gradients that no test
without a finite-difference reference would catch.

## Known-incorrect combination

!!! warning "Zygote + ChordalGMRF"
    Differentiating `logpdf` or `gaussian_approximation` through a `ChordalGMRF`
    with Zygote returns gradients that are wrong by roughly 10–15% when the
    precision matrix has Cholesky fill-in, with no error and no warning. Use
    Mooncake (the backend `ChordalGMRF` is designed for) or ForwardDiff instead.

## Enzyme

Enzyme differentiates the LLVM IR of whatever it is given, so anything it can
reach it will attempt — including CHOLMOD's `ccall`s, which carry no derivative
information, and the pure-Julia multifrontal factorization behind `ChordalGMRF`,
which it walks and gets wrong. Every operation that consumes a factorization
therefore needs an explicit rule, and the package refuses the operations it has no
rule for.

Supported: construction of `GMRF` and `WorkspaceGMRF`, `logdetcov`, and `logpdf`,
for `GMRF`, `ChordalGMRF`, `WorkspaceGMRF` and `MetaGMRF` priors.

Not supported, raising `ArgumentError`:

- `gaussian_approximation` for any prior. Its Implicit-Function-Theorem rule has to
  move precision cotangents between the prior and the approximation posterior,
  which store precisions under different conventions; the previous implementation
  got this wrong silently. Use Zygote, Mooncake or ForwardDiff.
- `var` and `std`. Their tangent is `-(Q⁻¹ Q̇ Q⁻¹)ᵢᵢ`, which needs full rows of
  `Q⁻¹` rather than the selected inverse, so there is no sparse rule to write.
- `WorkspaceGMRF` with linear equality constraints, whose `logpdf` carries a
  constraint-correction term that depends on `Q`.

### Specify the linear solver explicitly

The two-argument `GMRF` constructor picks a solver by runtime dispatch on the
precision matrix type, which is not type-stable and upsets Enzyme. Always name the
algorithm:

```julia
gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())  # general sparse
gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())     # SymTridiagonal (AR1, RW1)
gmrf = GMRF(μ, Q, LinearSolve.DiagonalFactorization()) # diagonal (IID)
```

### Julia version differences

Enzyme's behaviour varies by Julia version even with identical package versions,
so the supported set was verified separately on 1.10.10, 1.11.9 and 1.12.6.
Julia 1.10 and 1.12 behave identically across every operation and GMRF type.
Julia 1.11 has one difference:

!!! note "ChordalGMRF is not differentiable with Enzyme on Julia 1.11"
    `logdetcov` and `logpdf` on a `ChordalGMRF` work under Enzyme on Julia 1.10
    and 1.12, but fail with `EnzymeNoDerivativeError` on 1.11. Use ForwardDiff or
    Mooncake there, or run the pipeline on a `GMRF`/`WorkspaceGMRF` instead. This
    is a loud failure, not a wrong answer.

The *error type* on unsupported paths also varies. Where this package's own rule
refuses first you get an `ArgumentError` naming the operation and type; where
Enzyme's compilation fails earlier you get an Enzyme error. `var` on a `GMRF` is
one of the latter on Julia 1.11 and 1.12, where Enzyme's type analysis gives up
on `sum(::SparseVector)` before rule dispatch. Both outcomes are errors, never
wrong numbers.

## ForwardDiff

ForwardDiff has Dual-aware paths for every GMRF type. The one exception is
`var`/`std`, which raises: the LinearSolve cache behind a `GMRF{<:Dual}` is built
from primal data, so differentiating marginal variances used to return an exactly
zero gradient rather than an error.

Second-order derivatives via nested Duals are not supported. For Hessians, use
finite differences over a ForwardDiff gradient
(`FiniteDiff.finite_difference_jacobian` of `ForwardDiff.gradient`), which is more
stable than a pure finite-difference Hessian.

## Constrained GMRFs

`ConstrainedGMRF` priors (from `RW1Model`, `BesagModel`, …) are supported by
Zygote and ForwardDiff. Enzyme has no `ConstrainedGMRF` rules and will refuse.

## Testing your own gradients

If you add a model or an observation likelihood, check its gradients against
finite differences using a precision matrix **with Cholesky fill-in** — a 2D grid
Laplacian, not a tridiagonal AR(1) precision. Several of the bugs behind the table
above survived for a long time precisely because the tests used AR(1) precisions,
whose factor has no fill-in and so cannot exercise the code path where a selected
inverse is written back onto `Q`'s sparsity pattern.

## See Also

- [Automatic Differentiation for GMRF Hyperparameters](@ref): practical examples
- [ChainRulesCore.jl docs](https://juliadiff.org/ChainRulesCore.jl/stable/)
- [Enzyme.jl docs](https://enzyme.mit.edu/julia/stable/)
