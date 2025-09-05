using SparseArrays
using LibGEOS

export contiguity_adjacency

"""
    contiguity_adjacency(geoms; rule = :queen)

Build a binary contiguity adjacency matrix `W` for polygonal geometries.

- `geoms`: Vector of geometries. Accepts `LibGEOS.Geometry` or GeoInterface-compatible objects
           that LibGEOS can wrap via `LibGEOS.Geometry(geom)`.
- `rule`:  Contiguity definition. Currently supports `:queen` (share any boundary point).

Returns a sparse, symmetric `SparseMatrixCSC{Float64,Int}` with zero diagonal, where
`W[i,j] = 1.0` iff areas i and j are contiguous under the chosen rule.

Notes
- Complexity is O(n²) over the number of polygons. For large `n`, consider spatial
  indexing or pre-filtering by bounding boxes before calling this utility.
- Rook contiguity (shared edge only) can be added later; `:queen` covers most use cases.
"""
function contiguity_adjacency(geoms; rule::Symbol = :queen)
    rule === :queen || throw(ArgumentError("rule must be :queen (rook support can be added later)"))

    n = length(geoms)
    n > 0 || throw(ArgumentError("geoms must be a non-empty collection"))

    # Convert to LibGEOS geometries and prepare for fast touch tests
    g = Vector{LibGEOS.Geometry}(undef, n)
    prep = Vector{Any}(undef, n)
    for i in 1:n
        gi = geoms[i]
        g[i] = gi isa LibGEOS.Geometry ? gi : LibGEOS.Geometry(gi)
        prep[i] = LibGEOS.prepareGeom(g[i])
    end

    I = Int[]
    J = Int[]
    V = Float64[]

    @inbounds for i in 1:(n - 1)
        pi = prep[i]
        for j in (i + 1):n
            # Queen contiguity: boundaries touch (at least a point), interiors do not overlap
            if LibGEOS.preptouches(pi, g[j])
                push!(I, i); push!(J, j); push!(V, 1.0)
                push!(I, j); push!(J, i); push!(V, 1.0)
            end
        end
    end

    return sparse(I, J, V, n, n)
end

"""
    contiguity_adjacency(gc::LibGEOS.GeometryCollection; rule = :queen)

Build contiguity adjacency from a LibGEOS `GeometryCollection` by extracting its
member geometries via `LibGEOS.getGeometries(gc)` and delegating to
`contiguity_adjacency(::Vector)`.
"""
function contiguity_adjacency(gc::LibGEOS.GeometryCollection; rule::Symbol = :queen)
    geoms = LibGEOS.getGeometries(gc)
    return contiguity_adjacency(geoms; rule = rule)
end
