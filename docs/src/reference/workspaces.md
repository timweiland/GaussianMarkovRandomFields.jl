# Workspaces

Workspaces let you reuse a symbolic Cholesky factorization across many
precision-matrix updates with a shared sparsity pattern. This is the key
optimization for inner loops that evaluate a GMRF at many hyperparameter
values — INLA-style hyperparameter optimization, TMB-style marginal-likelihood
maximization, HMC/NUTS over hyperparameters, or any MAP loop that repeatedly
re-factors the same model with different numeric values.

## Why workspaces exist

A fresh `GMRF` constructed from a sparse precision matrix `Q` pays the full
symbolic Cholesky cost every time: fill-reducing reordering, elimination
tree, supernode layout. For a typical `n = 10^4` SPDE-discretized Matérn
prior, that's milliseconds per call — negligible for one evaluation, but
dominant inside an inner loop that touches the prior hundreds or thousands
of times with the same sparsity pattern and only different nonzero values.

A workspace holds the symbolic analysis persistently and offers a
numeric-only refactorization on each update. The factorization pattern is
computed once; subsequent updates just re-run the numeric phase.

```julia
using GaussianMarkovRandomFields

model = AR1Model(1000)

# One-time symbolic factorization, stored in the workspace
ws = make_workspace(model; τ = 1.0, ρ = 0.5)

# Inner loop: each call does numeric-only refactorization
for θ in θ_grid
    gmrf = model(ws; θ...)   # WorkspaceGMRF<: AbstractGMRF
    ℓ = logpdf(gmrf, z)
    # ...
end
```

See also the workspace-reuse tutorial for a concrete timing comparison.

## Core types

- [`GMRFWorkspace`](@ref) — the default workspace; wraps a single sparse
  precision matrix and a [`WorkspaceBackend`](@ref) (CHOLMOD or
  CliqueTrees).
- [`WorkspaceGMRF`](@ref) — an `AbstractGMRF` backed by a `GMRFWorkspace`.
  Owns its own `nzval` snapshot and a version tag so multiple instances
  can safely share one workspace (switching between them triggers a
  numeric refactorization).
- [`WorkspacePool`](@ref) — a task-safe pool of workspaces for parallel
  use across Julia tasks.

## Parallelism and pools

[`WorkspacePool`](@ref) hands out workspaces via `Channel`-based
checkout/checkin, safe across tasks that may migrate threads:

```julia
pool = make_workspace_pool(model; size = Threads.nthreads())

Threads.@threads for θ in θ_grid
    with_workspace(pool) do ws
        gmrf = model(ws; θ...)
        # ...
    end
end
```

!!! note "CHOLMOD serializes factorizations"
    The default `CHOLMODBackend` uses CHOLMOD, which holds a global lock —
    concurrent factorizations serialize gracefully but don't run in
    parallel. For actual parallel numeric factorization, build the pool
    with `CliqueTreesBackend` workspaces (pure Julia, no global state).

## Extension protocol

Peer packages can provide their own workspace types for models that don't
fit `GMRFWorkspace`'s "one sparse precision, one factorization" shape —
for example, a model whose joint precision is a block structure across
several per-block factorizations rather than a single global factor.

!!! warning "Experimental API (pre v0.5)"
    The [`AbstractLatentWorkspace`](@ref) /
    [`AbstractLatentWorkspacePool`](@ref) protocol is experimental. Surface
    may shift until validated against at least two downstream
    implementations.

### Workspace protocol

A peer `AbstractLatentWorkspace` must support being passed as the second
argument to the model's callable form:

```julia
struct MyPeerWorkspace <: AbstractLatentWorkspace
    # internals
end

function (m::MyLatentModel)(ws::MyPeerWorkspace; kwargs...)
    # produce an AbstractGMRF using ws, with numeric refactorization
    # instead of fresh symbolic analysis
end

GaussianMarkovRandomFields.make_workspace(m::MyLatentModel; kwargs...) =
    MyPeerWorkspace(m; kwargs...)
```

How the workspace detects staleness or manages sharing across multiple
GMRFs referencing it is the implementation's own business.
`GMRFWorkspace` uses integer version tags (`loaded_version` +
`WorkspaceGMRF.version`), but this is *not* part of the protocol — peer
workspaces may use a different mechanism or sidestep sharing entirely.

### Pool protocol

A peer `AbstractLatentWorkspacePool` must implement [`checkout`](@ref) and
[`checkin`](@ref). The RAII-style [`with_workspace`](@ref) method is
provided by a generic default on the abstract pool type and does not need
to be reimplemented, though subtypes may override it to add logging,
metrics, or alternate resource semantics.

## API reference

### Abstract types

```@docs
AbstractLatentWorkspace
AbstractLatentWorkspacePool
```

### Factory hooks

```@docs
make_workspace
make_workspace_pool
```

### GMRFWorkspace

```@docs
GMRFWorkspace
update_precision!
update_precision_values!
ensure_numeric!
ensure_selinv!
workspace_solve
selinv
selinv_diag
backward_solve
dimension
```

### WorkspaceGMRF

```@docs
WorkspaceGMRF
ConstraintInfo
```

### WorkspacePool

```@docs
WorkspacePool
checkout
checkin
with_workspace
```

### Backends

```@docs
WorkspaceBackend
CHOLMODBackend
CliqueTreesBackend
```

## See Also

- [Latent Models](@ref) — the `LatentModel` interface that factory hooks
  dispatch on.
- [Automatic Differentiation](@ref) — `WorkspaceGMRF` supports Zygote
  rrules for `logpdf` and `gaussian_approximation`, and ForwardDiff
  through the unconstrained constructor.
