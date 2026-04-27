export AbstractLatentWorkspace, AbstractLatentWorkspacePool

"""
    AbstractLatentWorkspace

Abstract supertype for workspace objects that persist factorization state
across precision-matrix updates for a `LatentModel`.

!!! warning "Experimental API (pre v0.5)"
    The `AbstractLatentWorkspace` protocol is experimental. Surface may shift
    until validated against at least two downstream implementations.

# Protocol
Concrete subtypes must support the `(model::LatentModel)(ws; θ...)` callable
form, returning a valid `AbstractGMRF`. How the workspace detects staleness
or manages sharing across multiple GMRFs referencing it is an implementation
detail — e.g. `GMRFWorkspace` uses integer version tags, but peer workspaces
are free to use other mechanisms.

See also: [`make_workspace`](@ref), [`GMRFWorkspace`](@ref).
"""
abstract type AbstractLatentWorkspace end

"""
    AbstractLatentWorkspacePool

Abstract supertype for task-safe pools of workspaces for use across
multiple concurrent Julia tasks.

!!! warning "Experimental API (pre v0.5)"
    See [`AbstractLatentWorkspace`](@ref) for stability status.

# Protocol
Concrete subtypes must implement [`checkout`](@ref) and [`checkin`](@ref).
A generic [`with_workspace`](@ref) default is provided on top of these;
subtypes may override it to add logging, metrics, or alternate resource
semantics.

See also: [`make_workspace_pool`](@ref), [`WorkspacePool`](@ref).
"""
abstract type AbstractLatentWorkspacePool end

include("backend.jl")
include("gmrf_workspace.jl")
include("cliquetrees_backend.jl")
include("workspace_gmrf.jl")
# gaussian_approximation.jl is included separately in GaussianMarkovRandomFields.jl
# because it depends on observation_models and arithmetic which are loaded later
