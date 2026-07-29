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
- **Enzyme.jl** for `logpdf`, `logdetcov` and `gaussian_approximation`. Fast, but
  it has the most caveats: no `var`/`std`, and constrained models need Julia 1.12.

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
| `gaussian_approximation` (`GMRF`) | ✅ | ✅ | ✅ |
| `logdetcov(::ChordalGMRF)` | ✅ | ❌ raises | ✅ except Julia 1.11 |
| `logpdf(::ChordalGMRF, z)` | ✅ | ⚠️ **silently wrong** | ✅ except Julia 1.11 |
| `gaussian_approximation` (`ChordalGMRF`) | ✅ | ⚠️ **silently wrong** | ✅ except Julia 1.11 |
| `logdetcov(::WorkspaceGMRF)` | ✅ | ❌ raises | ✅ |
| `logpdf(::WorkspaceGMRF, z)` | ✅ | ✅ | ✅ |
| `gaussian_approximation` (`WorkspaceGMRF`) | ✅ | ✅ | ✅ |
| `logpdf(::ConstrainedGMRF, z)` | ✅ | ✅ | Julia 1.12 only |
| `gaussian_approximation` (`ConstrainedGMRF`) | ✅ | ✅ | Julia 1.12 only |
| `var` / `std` (any type) | ✅ | ❌ raises | ❌ raises |

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

Supported: `logdetcov`, `logpdf`, and `gaussian_approximation`, for `GMRF`,
`ChordalGMRF`, `WorkspaceGMRF` and `MetaGMRF` priors, plus `ConstrainedGMRF` on
Julia 1.12 (see below). Gradients flow to both the prior's hyperparameters and
the observation likelihood's.

`gaussian_approximation` is differentiated with the Implicit Function Theorem
rather than through the Fisher-scoring loop, matching the backend-agnostic rule
the other backends use.

Not supported, raising `ArgumentError`:

- `var` and `std`. Their tangent is `-(Q⁻¹ Q̇ Q⁻¹)ᵢᵢ`, which reaches entries of
  `Q⁻¹` outside the selected-inverse pattern. ForwardDiff handles this by redoing
  the selected inversion in Dual arithmetic, which reverse mode has no equivalent
  of, so use ForwardDiff for marginal-variance gradients.
- `WorkspaceGMRF` with linear equality constraints, whose `logpdf` carries a
  constraint-correction term that depends on `Q`. Use `ConstrainedGMRF` instead.

!!! note "Constrained GMRFs need Julia 1.12 under Enzyme"
    `ConstrainedGMRF` stores a bare `Float64` (`log_constraint_correction`)
    alongside its array fields, which makes it a *mixed activity* type. Enzyme
    accepts such a type back from a custom rule on Julia 1.12, but on 1.10 and
    1.11 it raises
    `MixedReturnException: ... has mixed internal activity types. This is not
    presently supported.` — an upstream limitation, raised before any of this
    package's rules run. ForwardDiff and Zygote handle constrained models on
    every version.

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
Unconstrained `GMRF` and `WorkspaceGMRF` pipelines behave identically on all
three. Two differences remain, both of them loud failures rather than wrong
answers:

!!! note "ChordalGMRF is not differentiable with Enzyme on Julia 1.11"
    `logdetcov`, `logpdf` and `gaussian_approximation` on a `ChordalGMRF` work
    under Enzyme on Julia 1.10 and 1.12, but fail with `EnzymeNoDerivativeError`
    on 1.11. Use ForwardDiff or Mooncake there, or run the pipeline on a
    `GMRF`/`WorkspaceGMRF` instead.

The second is the `ConstrainedGMRF` mixed-activity limitation described above,
which excludes Julia 1.10 and 1.11.

The *error type* on unsupported paths can also vary: where this package's own
rule refuses first you get an `ArgumentError` naming the operation and type,
whereas if Enzyme's own compilation gives up earlier you get an Enzyme error
instead. Both are errors, never wrong numbers.

## ForwardDiff

ForwardDiff has Dual-aware paths for every GMRF type and every operation in the
table above.

One caveat on `var`/`std`: the GMRF must be built with a solver that supports
selected inversion (`CHOLMODFactorization`, `CholeskyFactorization`, …). Without
one, `var` falls back to the stochastic `RBMCStrategy` estimator, whose solves go
through the primal factorization and would drop every partial, so it raises
instead.

Second-order derivatives via nested Duals are not supported. For Hessians, use
finite differences over a ForwardDiff gradient
(`FiniteDiff.finite_difference_jacobian` of `ForwardDiff.gradient`), which is more
stable than a pure finite-difference Hessian.

## Constrained GMRFs

`ConstrainedGMRF` priors (from `RW1Model`, `BesagModel`, …) are supported by
Zygote, ForwardDiff, and — on Julia 1.12 only — Enzyme. See the note in the
Enzyme section for why the older versions are excluded.

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
