################################################################################
#    Barrier Matérn model (Bakka et al., 2019)
#
#    Non-stationary ν = 1 (α = 2) Matérn SPDE in which correlation does not flow
#    across designated "barrier" triangles. Barrier triangles get a small fixed
#    range, so the field decorrelates sharply across them.
#
#    Precision (unscaled by τ):
#        Q = (2/π) · Aᵀ C̃⁻¹ A,
#    where, with per-triangle ranges r_k (r_1 = range outside, r_2 = barrier
#    range inside),
#        A   = diag(C) + Σ_k (r_k²/8) G_k      (C = full lumped mass)
#        C̃   = diag(Σ_k r_k² c_k)              (range²-weighted lumped mass)
#        G_k = stiffness assembled over region k's triangles
#        c_k = lumped mass restricted to region k's triangles
#    With a single uniform range this reduces exactly to the stationary ν = 1
#    Matérn precision `MaternModel(disc; smoothness = 0)` (the 2/π factor matches
#    that model's σ²-normalisation `1/(4πκ²)` once `κ² = 8/range²`).
################################################################################

"""
    assemble_barrier_fem(disc::FEMDiscretization{2}, barrier_cells)

Assemble the range-independent FEM matrices for the barrier model, split into the
*normal* region (region 1) and the *barrier* region (region 2 — the triangles
whose ids are in `barrier_cells`). Returns a `NamedTuple`:

- `C`: full lumped mass vector `c_i = ∫ φ_i` (`= c_normal + c_barrier`).
- `G_regions = (G_normal, G_barrier)`: stiffness `∫_{Ω_k} ∇φ_i·∇φ_j` per region
  (`G_normal + G_barrier =` full stiffness).
- `c_regions = (c_normal, c_barrier)`: lumped mass `∫_{Ω_k} φ_i` per region.
- `Q_pattern = (; colptr, rowval)`: the range-invariant structural sparsity
  pattern of the precision `Aᵀ C̃⁻¹ A`; every assembly is padded to it.
"""
function assemble_barrier_fem(disc::FEMDiscretization{D}, barrier_cells) where {D}
    D == 2 || throw(ArgumentError("BarrierModel currently supports 2D discretizations only"))
    dh = disc.dof_handler
    interpolation = disc.interpolation
    cellvalues = CellValues(disc.quadrature_rule, interpolation, disc.geom_interpolation)
    Tv = Float64
    barrier_set = Set{Int}(barrier_cells)

    G_normal = allocate_matrix(SparseMatrixCSC{Tv, Int}, dh)
    G_barrier = allocate_matrix(SparseMatrixCSC{Tv, Int}, dh)
    M_normal = allocate_matrix(SparseMatrixCSC{Tv, Int}, dh)
    M_barrier = allocate_matrix(SparseMatrixCSC{Tv, Int}, dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ce = spzeros(Tv, n_basefuncs, n_basefuncs)
    Ge = spzeros(Tv, n_basefuncs, n_basefuncs)
    diffusion_factor = Matrix{Tv}(I, D, D)

    Gn_asm = start_assemble(G_normal)
    Gb_asm = start_assemble(G_barrier)
    Mn_asm = start_assemble(M_normal)
    Mb_asm = start_assemble(M_barrier)

    for (cell_idx, cell) in enumerate(CellIterator(dh))
        Ferrite.reinit!(cellvalues, cell)
        Ce = assemble_mass_matrix(Ce, cellvalues, interpolation; lumping = false)
        Ge = assemble_diffusion_matrix(Ge, cellvalues; diffusion_factor = diffusion_factor)
        dofs = celldofs(cell)
        if cell_idx in barrier_set
            assemble!(Gb_asm, dofs, Ge)
            assemble!(Mb_asm, dofs, Ce)
        else
            assemble!(Gn_asm, dofs, Ge)
            assemble!(Mn_asm, dofs, Ce)
        end
    end

    # Lumped mass per region = row sums of the consistent mass (∫_{Ω_k} φ_i).
    c_normal = Vector{Tv}(vec(sum(M_normal, dims = 2)))
    c_barrier = Vector{Tv}(vec(sum(M_barrier, dims = 2)))
    C = c_normal .+ c_barrier

    # Range-invariant structural pattern of Aᵀ C̃⁻¹ A, where A carries the union
    # pattern diag ∪ G_normal ∪ G_barrier at every range. All-ones values keep
    # every structurally reachable entry of the product strictly positive, so
    # sparse arithmetic cannot drop any of them as numerical zeros; the numeric
    # product's stored pattern at any range is a subset of this pattern.
    S = spdiagm(0 => ones(Tv, length(C))) + _ones_pattern(G_normal) + _ones_pattern(G_barrier)
    P = S' * S
    Q_pattern = (colptr = P.colptr, rowval = P.rowval)

    return (;
        C = C, G_regions = (G_normal, G_barrier),
        c_regions = (c_normal, c_barrier), Q_pattern = Q_pattern,
    )
end

# Factorization-free barrier precision (scaled by τ). Supports ForwardDiff.Dual
# τ/range since it never factorizes.
#
# The product's fill entries between second-order mesh neighbors can cancel to
# zero, and whether a given entry evaluates to exact 0.0 or to roundoff depends
# on the range. Sparse scalar `*` drops exact-zero stored entries, so scaling a
# raw sparse result makes the stored pattern range-dependent, which breaks
# fixed-pattern workspaces (issue #183). Instead, the result is scattered into
# the precomputed structural pattern with the τ·2/π scaling folded into the
# scatter, so the stored pattern is identical for every θ.
function _barrier_precision_only(fem, τ, range, range_fraction)
    (; C, G_regions, c_regions, Q_pattern) = fem
    r1 = range
    r2 = range_fraction * range
    Cdiag = r1^2 .* c_regions[1] .+ r2^2 .* c_regions[2]
    Cinv = spdiagm(0 => inv.(Cdiag))
    A = spdiagm(0 => C) + (r1^2 / 8) * G_regions[1] + (r2^2 / 8) * G_regions[2]
    return Symmetric(_pad_scaled_to_pattern(A' * Cinv * A, τ * (2 / π), Q_pattern))
end

"""
    BarrierModel(disc::FEMDiscretization; barrier_cells = Int[], range_fraction = 0.1,
                 alg = CHOLMODFactorization(), constraint = nothing,
                 observation_points = nothing)
    BarrierModel(disc, barrier_cells; kwargs...)

Construct a barrier Matérn latent model on a 2D FEM discretization. `barrier_cells`
are the triangle ids that act as barriers; `range_fraction ∈ (0, 1)` sets the
barrier range as a fraction of the normal `range` (Bakka et al. use ≈ 0.1).

Use [`barrier_triangles`](@ref) to obtain `barrier_cells` from a barrier polygon.
With no barrier triangles the model reduces to the stationary ν = 1 Matérn.
"""
function BarrierModel(
        discretization::F;
        barrier_cells::AbstractVector{<:Integer} = Int[],
        range_fraction::Real = 0.1,
        alg = CHOLMODFactorization(),
        constraint = nothing,
        observation_points = nothing,
    ) where {F <: FEMDiscretization}
    (0 < range_fraction < 1) ||
        throw(ArgumentError("range_fraction must be in (0, 1), got range_fraction=$range_fraction"))
    barrier_vec = collect(Int, barrier_cells)
    n = ndofs(discretization)
    processed_constraint = _process_constraint(constraint, n)
    fem_matrices = assemble_barrier_fem(discretization, barrier_vec)
    return BarrierModel{
        F, typeof(alg), typeof(processed_constraint),
        typeof(observation_points), typeof(fem_matrices),
    }(
        discretization, Float64(range_fraction), barrier_vec, alg,
        processed_constraint, observation_points, fem_matrices,
    )
end

function BarrierModel(discretization::FEMDiscretization, barrier_cells::AbstractVector{<:Integer}; kwargs...)
    return BarrierModel(discretization; barrier_cells = barrier_cells, kwargs...)
end

Base.length(model::BarrierModel) = ndofs(model.discretization)

hyperparameters(::BarrierModel) = (τ = Real, range = Real)

function _validate_barrier_parameters(; τ::Real, range::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    range > 0 || throw(ArgumentError("Range parameter must be positive, got range=$range"))
    return nothing
end

function precision_matrix(model::BarrierModel; τ::Real, range::Real, kwargs...)
    _validate_barrier_parameters(; τ = τ, range = range)
    return _barrier_precision_only(model.fem_matrices, τ, range, model.range_fraction)
end

mean(model::BarrierModel; kwargs...) = zeros(length(model))

constraints(model::BarrierModel; kwargs...) = model.constraint

model_name(::BarrierModel) = :barrier

# Prediction operator, mirroring `MaternModel`.
function evaluation_matrix(model::BarrierModel, points::AbstractMatrix)
    return evaluation_matrix(model.discretization, points)
end

function evaluation_matrix(model::BarrierModel)
    model.observation_points === nothing && throw(
        ArgumentError(
            "No observation points stored. Use `evaluation_matrix(model, points)` " *
                "with explicit points, or construct the model with `observation_points`.",
        ),
    )
    return evaluation_matrix(model.discretization, model.observation_points)
end

################################################################################
#    Barrier-triangle selection
################################################################################

# Normalize a polygon spec to a vector of (x, y) tuples.
_polygon_points(poly::AbstractMatrix) = [(Float64(poly[i, 1]), Float64(poly[i, 2])) for i in 1:size(poly, 1)]
_polygon_points(poly::AbstractVector) = [(Float64(p[1]), Float64(p[2])) for p in poly]

# Ray-casting point-in-polygon test for a simple polygon.
function _point_in_polygon(px, py, poly)
    inside = false
    n = length(poly)
    j = n
    for i in 1:n
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    barrier_triangles(disc::FEMDiscretization{2}, polygon) -> Vector{Int}

Return the ids of mesh triangles whose **centroid** lies inside `polygon`, the
standard rule for tagging barrier triangles (Bakka et al., 2019). `polygon` is a
closed simple polygon given either as an `N×2` matrix of vertices or as a vector
of points (`(x, y)` tuples or `Vec`s). The result can be passed straight to
[`BarrierModel`](@ref) as `barrier_cells`.
"""
function barrier_triangles(disc::FEMDiscretization{D}, polygon) where {D}
    D == 2 || throw(ArgumentError("barrier_triangles supports 2D discretizations only"))
    poly = _polygon_points(polygon)
    grid = disc.grid
    cells = grid.cells
    nodes = grid.nodes
    ids = Int[]
    for cid in eachindex(cells)
        nodeids = cells[cid].nodes
        cx = 0.0
        cy = 0.0
        for nid in nodeids
            x = nodes[nid].x
            cx += x[1]
            cy += x[2]
        end
        m = length(nodeids)
        if _point_in_polygon(cx / m, cy / m, poly)
            push!(ids, cid)
        end
    end
    return ids
end
