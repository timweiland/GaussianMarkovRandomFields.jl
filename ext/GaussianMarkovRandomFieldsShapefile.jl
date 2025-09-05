module GaussianMarkovRandomFieldsShapefile

using GaussianMarkovRandomFields
using Shapefile
using GeoInterface
using LibGEOS

"""
    contiguity_adjacency(path::AbstractString; rule = :queen, id_field::Union{Nothing,Symbol} = nothing)

Read a shapefile and build a binary contiguity adjacency matrix `W` for its polygon features.

Arguments
- `path`: Path to a `.shp` file (the corresponding `.dbf` must be present alongside).
- `rule`: Contiguity definition (`:queen` supported).
- `id_field`: Optional attribute field (Symbol) to return feature identifiers alongside `W`.

Returns
- `(W, ids)`: `W::SparseMatrixCSC{Float64,Int}` and a vector `ids` with either the chosen
  `id_field` or `1:n` if `id_field === nothing`.

Notes
- Geometries are extracted in file order. Join your tabular data with the returned `ids`
  to ensure consistent indexing with the adjacency matrix.
"""
function GaussianMarkovRandomFields.contiguity_adjacency(
        path::AbstractString; rule::Symbol = :queen, id_field::Union{Nothing, Symbol} = nothing
    )
    table = Shapefile.Table(path)
    return GaussianMarkovRandomFields.contiguity_adjacency(table; rule, id_field)
end

"""
    contiguity_adjacency(table::Shapefile.Table; rule = :queen, id_field::Union{Nothing,Symbol} = nothing)

Variant that accepts an already-open `Shapefile.Table`.
"""
function GaussianMarkovRandomFields.contiguity_adjacency(
        table::Shapefile.Table; rule::Symbol = :queen, id_field::Union{Nothing, Symbol} = nothing
    )
    geoms = GeoInterface.convert.(Ref(LibGEOS), table.geometry)
    W = GaussianMarkovRandomFields.contiguity_adjacency(geoms; rule = rule)
    node_to_id = id_field === nothing ? collect(1:length(geoms)) : getproperty(table, id_field)
    id_to_node = Dict([(id, node) for (node, id) in enumerate(node_to_id)])
    return (W = W, node_to_id = node_to_id, id_to_node = id_to_node)
end

end
