"""
    auto_size_params(points; α::Real=0.8, β::Real=3.0, γ::Real=3.0)

Given `points` (vector of tuples / vectors), compute automatic sizing
parameters for the Gmsh `Threshold` field:

Returns `(sizeMin, sizeMax, distMin, distMax, interior_mesh_size)` where
`interior_mesh_size` is a convenient overall interior target (median
nearest-neighbour distance).
"""
function auto_size_params(points; α::Real = 0.8, β::Real = 3.0, γ::Real = 3.0)
    n = length(points)
    @assert n >= 2 "Need at least two points to compute spacing."

    d = length(points[1])
    data = zeros(d, n)
    for i in 1:n
        data[:, i] .= points[i]
    end

    tree = KDTree(data)
    _, dists = knn(tree, data, 2, true)
    d_i = [d[2] for d in dists]

    dmin = minimum(d_i)
    dmed = median(d_i)

    sizeMin = α * dmed
    sizeMax = β * dmed
    distMin = dmin
    distMax = γ * dmed
    interior_mesh_size = dmed

    return sizeMin, sizeMax, distMin, distMax, interior_mesh_size
end

"""
    _ring_xy(polygon::LibGEOS.Polygon)

Extract the exterior ring of a `LibGEOS.Polygon` as a `Vector` of
`(x, y)` `Tuple`s, dropping the duplicated closing vertex.
"""
function _ring_xy(polygon::LibGEOS.Polygon)
    ring = LibGEOS.exteriorRing(polygon)
    cs = LibGEOS.getCoordSeq(ring)
    n = LibGEOS.getSize(cs)
    pts = [(LibGEOS.getX(cs, i), LibGEOS.getY(cs, i)) for i in 1:n]
    if length(pts) >= 2 && pts[1] == pts[end]
        pop!(pts)
    end
    return pts
end

function _add_poly_to_gmsh(polygon::LibGEOS.Polygon, mesh_size; hole::Bool = false)
    ring = _ring_xy(polygon)
    # create gmsh points
    p = [gmsh.model.geo.addPoint(x, y, 0.0, mesh_size) for (x, y) in ring]
    # lines in consistent order
    lines = Int[]
    for i in 1:(length(p) - 1)
        push!(lines, gmsh.model.geo.addLine(p[i], p[i + 1]))
    end
    push!(lines, gmsh.model.geo.addLine(p[end], p[1]))  # close
    # inner loops (holes) must be opposite orientation
    if hole
        return gmsh.model.geo.addCurveLoop(reverse(lines))
    else
        return gmsh.model.geo.addCurveLoop(lines)
    end
end

function _setup_gmsh_common(model_name::String)
    Gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    return gmsh.model.add(model_name)
end

function _configure_mesh_fields(ps_inner, sizeMin, sizeMax, distMin, distMax)
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", ps_inner)

    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", sizeMin)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", sizeMax)
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", distMin)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", distMax)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_th)
    return f_th
end

function _finalize_mesh(element_order, save_path, dim)
    if element_order != 1
        gmsh.model.mesh.setOrder(element_order)
    end
    gmsh.model.mesh.reverse()

    facedim = dim - 1
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    nodes = FerriteGmsh.tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    domaincellset = cellsets["Domain"]
    elements = elements[collect(domaincellset)]

    boundarydict = toboundary(facedim)
    facetsets = tofacetsets(boundarydict, elements)

    if save_path !== nothing
        gmsh.write(save_path)  # COV_EXCL_LINE
    end
    gmsh.finalize()

    return Ferrite.Grid(elements, nodes, facetsets = facetsets, cellsets = cellsets)
end

function _create_physical_groups(dim, domain_entities, interior_entities)
    group = gmsh.model.addPhysicalGroup(dim, domain_entities)
    gmsh.model.setPhysicalName(dim, group, "Domain")
    int_group = gmsh.model.addPhysicalGroup(dim, interior_entities)
    gmsh.model.setPhysicalName(dim, int_group, "Interior")
    return group, int_group
end

@doc raw"""
    generate_mesh(points; element_order::Int=1, save_path=nothing)

Generate a `Ferrite.Grid` from a list of points using Gmsh.

`points` is an `AbstractVector` of point-like objects (tuples, vectors,
or anything iterable yielding 1 or 2 real numbers), or an `AbstractMatrix`
where each row is a point. The spatial dimension is inferred from the
length of the first point.

For 2D point clouds the algorithm is:

1. Build the convex hull and an inflated buffer polygon (LibGEOS).
2. Triangulate the buffered region with Gmsh, embedding the interior
   points as constrained vertices.
3. Transfer the Gmsh mesh into a `Ferrite.Grid` (via `FerriteGmsh`).

For 1D point clouds the analogous bracketed line segment is meshed.

# Keyword arguments
- `element_order`: order of the FEM elements (default: 1).
- `save_path`: optional path to save the Gmsh `.msh` file.
"""
function generate_mesh(
        points::AbstractVector;
        element_order::Int = 1,
        save_path = nothing,
    )
    n = length(points)
    n >= 2 || throw(ArgumentError("generate_mesh needs at least 2 points, got $n"))

    # Normalise each point to (x[, y]) Float64 Tuple
    pts = [_as_tuple(p) for p in points]
    d = length(pts[1])
    all(length(p) == d for p in pts) || throw(ArgumentError("all points must have the same dimension"))

    if d == 1
        return _generate_mesh_1d(pts; element_order = element_order, save_path = save_path)
    elseif d == 2
        return _generate_mesh_2d(pts; element_order = element_order, save_path = save_path)
    else
        throw(ArgumentError("generate_mesh only supports 1D and 2D point sets, got dimension $d"))  # COV_EXCL_LINE
    end
end

function generate_mesh(
        points::AbstractMatrix;
        element_order::Int = 1,
        save_path = nothing,
    )
    pts = [Tuple(Float64.(view(points, i, :))) for i in 1:size(points, 1)]
    return generate_mesh(pts; element_order = element_order, save_path = save_path)
end

_as_tuple(p::Tuple) = Tuple(Float64.(p))
_as_tuple(p::AbstractVector) = Tuple(Float64.(p))
_as_tuple(p) = Tuple(Float64.(collect(p)))

function _generate_mesh_1d(points::AbstractVector{<:Tuple}; element_order::Int, save_path)
    sizeMin, sizeMax, distMin, distMax, interior_mesh_size = auto_size_params(points)
    buffer_width = 1.5 * sizeMax

    coords = sort!([p[1] for p in points])
    min_x, max_x = extrema(coords)
    x_start = min_x - buffer_width
    x_end = max_x + buffer_width

    _setup_gmsh_common("Scattered mesh")

    p_int_1 = gmsh.model.geo.addPoint(min_x, 0, interior_mesh_size)
    p_int_2 = gmsh.model.geo.addPoint(max_x, 0, interior_mesh_size)
    p_ext_1 = gmsh.model.geo.addPoint(x_start, 0, interior_mesh_size)
    p_ext_2 = gmsh.model.geo.addPoint(x_end, 0, interior_mesh_size)
    l_int = gmsh.model.geo.addLine(p_int_1, p_int_2)
    l_ext = gmsh.model.geo.addLine(p_ext_1, p_ext_2)

    coord_vec = [p[1] for p in points]
    ch_points_inds = indexin([min_x, max_x], coord_vec)
    inner = points[1:end .∉ Ref(ch_points_inds)]
    ps_inner = [gmsh.model.geo.addPoint(p[1], 0, interior_mesh_size) for p in inner]

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, ps_inner, 1, l_int)

    _create_physical_groups(1, [l_ext, l_int], [l_int])
    _configure_mesh_fields(ps_inner, sizeMin, sizeMax, distMin, distMax)

    gmsh.model.mesh.generate(1)
    return _finalize_mesh(element_order, save_path, 1)
end

function _generate_mesh_2d(points::AbstractVector{<:Tuple}; element_order::Int, save_path)
    sizeMin, sizeMax, distMin, distMax, interior_mesh_size = auto_size_params(points)
    buffer_width = 1.5 * sizeMax

    mp = LibGEOS.MultiPoint([[x, y] for (x, y) in points])
    hull = LibGEOS.convexhull(mp)
    outer = LibGEOS.buffer(hull, buffer_width)

    _setup_gmsh_common("Scattered mesh")

    c_ext = _add_poly_to_gmsh(outer, sizeMax, hole = false)
    c_int = _add_poly_to_gmsh(hull, interior_mesh_size, hole = true)
    s_int = gmsh.model.geo.addPlaneSurface([c_int])
    s_ext = gmsh.model.geo.addPlaneSurface([c_ext, c_int])

    hull_boundary = LibGEOS.boundary(hull)
    inner = filter(p -> LibGEOS.distance(hull_boundary, LibGEOS.Point(p[1], p[2])) > 1.0e-8, points)
    ps_inner = [gmsh.model.geo.addPoint(p[1], p[2], 0, interior_mesh_size) for p in inner]

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, ps_inner, 2, s_int)

    _create_physical_groups(2, [s_ext, s_int], [s_int])
    _configure_mesh_fields(ps_inner, sizeMin, sizeMax, distMin, distMax)

    gmsh.model.mesh.generate(2)
    return _finalize_mesh(element_order, save_path, 2)
end
