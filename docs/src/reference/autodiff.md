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
- **Mooncake.jl** for anything built on the pure-Julia CliqueTrees
  factorization — a `GMRF` constructed with `CliqueTreesFactorization()`, a
  `WorkspaceGMRF` over a `CliqueTreesBackend`, `ChordalGMRF`, and constrained
  versions of all three. It is the only reverse-mode backend that also covers
  `var`/`std`, and its cost per gradient does not grow with the number of
  hyperparameters. It refuses a CHOLMOD-backed GMRF.
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
| `logdetcov(::GMRF)` | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `logpdf(::GMRF, z)` | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `gaussian_approximation` (`GMRF`) | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `logdetcov(::ChordalGMRF)` | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `logpdf(::ChordalGMRF, z)` | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `gaussian_approximation` (`ChordalGMRF`) | ✅ | ✅ | ⚠️ unreliable | ✅ |
| `logdetcov(::WorkspaceGMRF)` | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `logpdf(::WorkspaceGMRF, z)` | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `gaussian_approximation` (`WorkspaceGMRF`) | ✅ | ✅ | ✅ | ✅ CliqueTrees only |
| `logpdf(::ConstrainedGMRF, z)` | ✅ | ✅ | Julia 1.12 only | ✅ CliqueTrees only |
| `gaussian_approximation` (`ConstrainedGMRF`) | ✅ | ✅ | Julia 1.12 only | ✅ CliqueTrees only |
| `var` / `std` (`GMRF`, `WorkspaceGMRF`) | ✅ | ❌ raises | ❌ raises | ✅ CliqueTrees only |
| `var` / `std` (`ChordalGMRF`) | ✅ | ❌ raises | ❌ raises | ✅ |

✅ matches finite differences · ❌ raises an error · ⚠️ see the note for that row

"CliqueTrees only" means the Mooncake rules need the pure-Julia CliqueTrees
factorization rather than CHOLMOD: build the GMRF with
`CliqueTreesFactorization()`, or the workspace with `CliqueTreesBackend`. On a
CHOLMOD-backed one, Mooncake raises. See [Mooncake](#Mooncake) below.

The Enzyme `ChordalGMRF` rows depend on the Julia version *and* the CPU
architecture; see the note under [Enzyme](#Enzyme) below. The verification above
ran on aarch64.

Every ❌ in this table is an explicit, actionable error. Combinations that cannot
be supported raise rather than falling back to differentiating a sparse
factorization, because doing the latter produces wrong gradients that no test
without a finite-difference reference would catch.

## Zygote

Zygote works through the package's ChainRules rules, which cover `logdetcov`,
`logpdf` and `gaussian_approximation` for every GMRF type in the table. Each one
uses a selected inverse rather than differentiating the sparse factorization.

The workspace reuse path — `make_workspace(model; θ...)` once, then
`model(ws; θ...)` inside the differentiated function — also works under Zygote.
The callable carries an rrule that reruns the model's `mean` and
`precision_matrix` construction in reverse mode while keeping the workspace
itself (including its `update_precision!` mutation) out of the trace.
`SeparableModel` is the exception: its Kronecker-product construction is not
yet reverse-mode differentiable, so use ForwardDiff for separable models.

`var` and `std` raise an `ArgumentError`. Their tangent needs entries of the
covariance *outside* the precision's sparsity pattern, which selected inversion
does not compute, so there is no cheap rule to write and one built on `selinv`
would be silently wrong. Use ForwardDiff for marginal-variance gradients.

The contrast with `logdetcov` is the whole reason that one is supported:
`∂logdetcov/∂Q = -Q⁻¹` is only ever contracted against a `dQ` living on `Q`'s own
pattern, so the selected inverse is exact there, whereas `∂diag(Q⁻¹)/∂Q` is not.

## Mooncake

Mooncake needs a factorization it can traverse, so its support follows the
factorization rather than the GMRF type: everything built on **CliqueTrees.jl's
pure-Julia multifrontal Cholesky** is supported, and CHOLMOD is refused.

Three constructions get you there, and all of them support `logdetcov`,
`logpdf`, `var`/`std` and `gaussian_approximation`, constrained or not:

```julia
using GaussianMarkovRandomFields, LinearSolve
using DifferentiationInterface, Mooncake, MooncakeSparse

x = GMRF(μ, Q, CliqueTreesFactorization())   # a standard GMRF, chordal backend
x = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q_pattern, CliqueTreesBackend))
x = ChordalGMRF(μ, Q)                        # the factorization, unwrapped
```

`MooncakeSparse` must be loaded alongside `Mooncake`; the rules for sparse
precisions live there.

A typical hyperparameter objective, with the workspace built once outside the
differentiated function so its symbolic factorization is reused:

```julia
ws = GMRFWorkspace(Q_pattern, CliqueTreesBackend)

function objective(θ)
    prior = WorkspaceGMRF(build_mean(θ), build_precision(θ), ws)
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, y)
end

grad = DifferentiationInterface.gradient(objective, AutoMooncake(), θ)
```

Linear equality constraints (sum-to-zero and friends) differentiate on both
constrained paths — a `ConstrainedGMRF` over a CliqueTrees-backed `GMRF`, and a
constrained `WorkspaceGMRF` — including through `mean(posterior)`.

### Why reach for it

Mooncake is the only reverse-mode backend here that covers `var`/`std`, and the
only one whose cost is flat in the number of hyperparameters: a gradient of the
Laplace marginal likelihood runs at roughly 1.0–1.7× a primal evaluation
regardless of how many hyperparameters there are, where ForwardDiff scales
linearly in them. At n = 4096 with 32 hyperparameters that is about 38× faster
than ForwardDiff and 20× faster than Zygote on the same objective. With a
handful of hyperparameters ForwardDiff is still the simpler choice.

`gaussian_approximation` is differentiated with the Implicit Function Theorem —
one differentiable Newton step at the converged mode — rather than through the
Fisher-scoring loop, matching the rule the other backends use.

### What raises

A CHOLMOD-backed GMRF — what the two-argument constructor resolves to for a
general sparse precision — raises an `ArgumentError` naming the construction to
use instead. This is a guard, not an oversight: Mooncake reaches CHOLMOD only as
pointers into memory owned by a C library, and left to itself it dereferences
one and the process dies with a segmentation fault, taking the rest of the
session with it. A `WorkspaceGMRF` over the default `CHOLMODBackend` raises the
same way.

The refusal fires on the operations that actually consume the factorization, not
on the presence of a GMRF; the parts that need none, such as the mean or the
quadratic form behind `sqmahal`, still differentiate normally.

Gauss–Newton (`NonlinearLeastSquares`) likelihoods raise an actionable error
under Mooncake, as they do under Zygote and Enzyme: their score needs a
forward-mode sparse Jacobian that reverse mode cannot differentiate through.
Use ForwardDiff for those.

As with Enzyme, the *error type* can vary. Where this package's guard refuses
first you get the `ArgumentError` naming the operation; where Mooncake's own
rule compiler gives up earlier you get a `MooncakeRuleCompilationError`, which
says nothing about GMRFs. Known cases of the second: `rand` on every version,
and on Julia 1.10 a CHOLMOD-backed GMRF over a precision denser than a
tridiagonal one. Both are loud failures — never a wrong number, and, which is
the point of the guard, never a crash.

!!! note "CliqueTrees 1.19.5 or newer for `var` gradients"
    Before 1.19.5, CliqueTrees' selected-inversion rule read the selected
    inverse already projected onto the precision's sparsity pattern, so entries
    on the factor's fill pattern but outside it were silently taken as zero and
    `var`/`std` gradients came out a few percent low. The error was exactly zero
    for precisions whose pattern is chordal — a tridiagonal AR(1) precision, for
    instance — which is what made it easy to miss. The package's compat bound
    requires the fixed version; `logdetcov` and `logpdf` gradients were correct
    throughout.

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
Zygote, ForwardDiff, Mooncake (over a CliqueTrees-backed base GMRF) and — on
Julia 1.12 only — Enzyme. See the note in the Enzyme section for why the older
versions are excluded. Constrained `WorkspaceGMRF`s are supported by the same
set except Enzyme, which has no rule for their `Q`-dependent constraint
correction.

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
