# Automatic Differentiation Reference

GaussianMarkovRandomFields.jl supports gradient-based inference and optimization
through several AD backends. Which backend to reach for depends on **both** the
operation and the GMRF type — support is not uniform, and the differences are
large enough to matter in practice.

## Two kinds of differentiation

It is worth separating two things that both get called "AD support", because the
answer is different for each.

**Differentiating through GMRF internals** — `logpdf`, `logdetcov`, `var` and
`gaussian_approximation`, with respect to hyperparameters — runs through a sparse
Cholesky factorization. No general-purpose AD system can traverse that: CHOLMOD is
a C library reached through opaque pointers, and even the pure-Julia multifrontal
factorization behind `ChordalGMRF` is walked incorrectly by backends that try. So
every supported combination here rests on a **hand-written rule**, and the support
matrix below records exactly which ones exist. This is the part where the choice of
backend genuinely constrains what you can do.

**Differentiating your own likelihood** — through
[`AutoDiffObservationModel`](@ref) and [`AutoDiffLatentPrior`](@ref) — is a
different situation. There the package differentiates *your* Julia function, not
its own linear algebra, and it does so through
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl).
Any DI-compatible backend can be passed as `grad_backend`; the package does not
need a rule for it. The default picks the first of Enzyme, Mooncake, Zygote or
ForwardDiff that is loaded, and you can override it:

```julia
using DifferentiationInterface, Enzyme
obs_model = AutoDiffObservationModel(my_loglik; n_latent = n, grad_backend = AutoEnzyme())
```

The support matrix constrains the first kind, not the second.

## Choosing a backend

**Start with ForwardDiff.jl.** It is the only backend that currently handles every
GMRF type and every differentiable operation in this package, and it is genuinely
fast for hyperparameter optimization, where the input is a handful of
hyperparameters and forward mode is the right shape for the problem.

Reach for a reverse-mode backend when the parameter vector is large enough that
forward mode's per-parameter cost dominates:

- **Zygote.jl** for `GMRF`, `ChordalGMRF` and `WorkspaceGMRF` priors.
- **Mooncake.jl** for `ChordalGMRF`, which is what the chordal backend is built
  for. It refuses a plain `GMRF`.
- **Enzyme.jl** for `logpdf`, `logdetcov` and `gaussian_approximation` on `GMRF`
  and `WorkspaceGMRF`. Fast, but it has the most caveats: no `var`/`std`,
  constrained models need Julia 1.12, and `ChordalGMRF` is unreliable.

## Support matrix

Every cell below was checked against finite differences on Julia 1.10.10, 1.11.9
and 1.12.6, using a precision matrix whose Cholesky factor has genuine fill-in.
The fill-in matters: a tridiagonal AR(1) precision has none, and several of these
failures are invisible with one.

| Operation | ForwardDiff | Zygote | Enzyme | Mooncake |
|---|---|---|---|---|
| `logdetcov(::GMRF)` | ✅ | ✅ | ✅ | ❌ raises |
| `logpdf(::GMRF, z)` | ✅ | ✅ | ✅ | ❌ raises |
| `gaussian_approximation` (`GMRF`) | ✅ | ✅ | ✅ | ❌ raises |
| `logdetcov(::ChordalGMRF)` | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `logpdf(::ChordalGMRF, z)` | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `gaussian_approximation` (`ChordalGMRF`) | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `logdetcov(::WorkspaceGMRF)` | ✅ | ✅ | ✅ | ❌ raises |
| `logpdf(::WorkspaceGMRF, z)` | ✅ | ✅ | ✅ | ❌ raises |
| `gaussian_approximation` (`WorkspaceGMRF`) | ✅ | ✅ | ✅ | ❌ raises |
| `logpdf(::ConstrainedGMRF, z)` | ✅ | ✅ | Julia 1.12 only | ❌ raises |
| `gaussian_approximation` (`ConstrainedGMRF`) | ✅ | ✅ | Julia 1.12 only | ❌ raises |
| `var` / `std` (`GMRF`, `WorkspaceGMRF`) | ✅ | ❌ raises | ❌ raises | ❌ raises |
| `var` / `std` (`ChordalGMRF`) | ✅ | ❌ raises | ❌ raises | ⚠️ wrong, see below |

✅ matches finite differences · ❌ raises an error · ⚠️ see the note for that row

The Enzyme `ChordalGMRF` rows depend on the Julia version *and* the CPU
architecture; see the note under [Enzyme](#Enzyme) below. The verification above
ran on aarch64.

The single ⚠️ is `var`/`std` on a `ChordalGMRF` under Mooncake, whose gradient is
currently around 1% off. The cause sits upstream of this package, in the
selected-inversion rule for the chordal factorization, and a fix is expected. Use
ForwardDiff for marginal-variance gradients in the meantime.

Everything else that is not ✅ raises an explicit, actionable error. Combinations
that cannot be supported refuse rather than falling back to differentiating a
sparse factorization, because doing the latter produces wrong gradients that no
test without a finite-difference reference would catch.

Every ❌ in this table is an explicit, actionable error. Combinations that cannot
be supported raise rather than falling back to differentiating a sparse
factorization, because doing the latter produces wrong gradients that no test
without a finite-difference reference would catch.

## Zygote

Zygote works through the package's ChainRules rules, which cover `logdetcov`,
`logpdf` and `gaussian_approximation` for every GMRF type in the table. Each one
uses a selected inverse rather than differentiating the sparse factorization.

`var` and `std` raise an `ArgumentError`. Their tangent needs entries of the
covariance *outside* the precision's sparsity pattern, which selected inversion
does not compute, so there is no cheap rule to write and one built on `selinv`
would be silently wrong. Use ForwardDiff for marginal-variance gradients.

The contrast with `logdetcov` is the whole reason that one is supported:
`∂logdetcov/∂Q = -Q⁻¹` is only ever contracted against a `dQ` living on `Q`'s own
pattern, so the selected inverse is exact there, whereas `∂diag(Q⁻¹)/∂Q` is not.

## Mooncake

Mooncake is built for `ChordalGMRF`, whose multifrontal factorization is pure
Julia and therefore something Mooncake can traverse. `logdetcov`, `logpdf` and
`gaussian_approximation` are all checked against finite differences on that type.

A `GMRF` backed by CHOLMOD — what the two-argument constructor resolves to for a
general sparse precision, and what `ConstrainedGMRF` wraps — raises an
`ArgumentError` instead. Mooncake reaches CHOLMOD only as pointers into memory
owned by a C library; left to itself it dereferences one and the process dies
with a segmentation fault, taking the rest of the session down with it. Refusing
is the only alternative, since there is nothing there for a generic AD to
traverse.

`WorkspaceGMRF` needs no such guard. It owns its `CHOLMOD.Factor` directly rather
than through the shared solver entry points, and Mooncake gives up on it with a
`TypeError` well before reaching the factor.

The refusal fires on the operations that actually consume the factorization —
`logdetcov` (and so `logpdf`), `var`/`std`, `selinv` and `backward_solve` — not
on the presence of a `GMRF`. The parts that need no factorization, such as the
mean or the quadratic form behind `sqmahal`, still differentiate normally.

As with Enzyme, the *error type* can vary. Where this package's guard refuses
first you get the `ArgumentError` naming the operation; where Mooncake's own rule
compiler gives up earlier you get a `MooncakeRuleCompilationError` instead, which
says nothing about GMRFs. Known cases of the second: `rand` on every version,
Julia 1.10 for precision matrices denser than a tridiagonal one, and
`gaussian_approximation` on x86-64 (where Mooncake stops at the Newton loop;
aarch64 reaches the guard and gives the actionable message). Both are loud
failures — never a wrong number, and, which is the point of the guard, never a
crash.

Use `ChordalGMRF` under Mooncake, or ForwardDiff, which handles every GMRF type.

## Enzyme

Enzyme differentiates the LLVM IR of whatever it is given, so anything it can
reach it will attempt — including CHOLMOD's `ccall`s, which carry no derivative
information, and the pure-Julia multifrontal factorization behind `ChordalGMRF`,
which it walks and gets wrong. Every operation that consumes a factorization
therefore needs an explicit rule, and the package refuses the operations it has no
rule for.

Supported: `logdetcov`, `logpdf`, and `gaussian_approximation`, for `GMRF`,
`WorkspaceGMRF` and `MetaGMRF` priors, plus `ConstrainedGMRF` on Julia 1.12 and
`ChordalGMRF` where Enzyme manages it (both noted below). Gradients flow to the
observation likelihood's hyperparameters as well as the prior's, and precisions
may be sparse, `Diagonal` or `SymTridiagonal`.

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
precision matrix type, so its return type infers as `Any`. Julia 1.12 copes with
that, but 1.10 and 1.11 reject the rule outright with `AugmentedRuleReturnError:
... a function which returned Any`. Always name the algorithm:

```julia
gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())  # general sparse
gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())     # SymTridiagonal (AR1, RW1)
gmrf = GMRF(μ, Q, LinearSolve.DiagonalFactorization()) # diagonal (IID)
```

### Julia version differences

Enzyme's behaviour varies by Julia version even with identical package versions,
so the supported set was verified separately on 1.10.10, 1.11.9 and 1.12.6.
Unconstrained `GMRF` and `WorkspaceGMRF` pipelines behave identically on all
three, including diagonal and `SymTridiagonal` precisions. Two caveats remain,
both of them loud failures rather than wrong answers:

!!! note "ChordalGMRF under Enzyme is unreliable"
    Whether Enzyme can differentiate a `ChordalGMRF` depends on the Julia
    version, the CPU architecture *and* the resolved package versions: the same
    Julia 1.12.6 succeeds on aarch64 and fails on x86-64 with `EnzymeNoTypeError`,
    and Julia 1.11 fails with `EnzymeNoDerivativeError`. When it runs it is
    correct — the package's rules are the same ones the other GMRF types use, and
    the test suite checks that a wrong answer is never returned — but it cannot be
    relied on. Use Mooncake (the backend `ChordalGMRF` is built for) or
    ForwardDiff, or run the pipeline on a `GMRF`/`WorkspaceGMRF`.

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
