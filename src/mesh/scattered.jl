using Gmsh, FerriteGmsh, Ferrite
using LibGEOS, GeoInterface
import GeometryBasics
using NearestNeighbors

export generate_mesh

"""
    auto_size_params(points; k::Int=6, α::Real=0.8, β::Real=3.0, γ::Real=3.0)

Given `points` (vector of (x,y) tuples), compute automatic sizing parameters
for Gmsh Threshold field:

Returns (sizeMin, sizeMax, distMin, distMax, interior_mesh_size)

`interior_mesh_size` is a convenient overall interior target (median kNN).
"""
function auto_size_params(points; α::Real = 1.0, β::Real = 3.0, γ::Real = 3.0)
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

    sizeMin = α * dmin
    sizeMax = β * dmed
    distMin = dmin
    distMax = γ * dmed
    interior_mesh_size = dmed

    return sizeMin, sizeMax, distMin, distMax, interior_mesh_size
end

function _add_poly_to_gmsh(poly, mesh_size)
    p_initial = gmsh.model.geo.addPoint(poly.exterior[1][1]..., 0, mesh_size)
    p_last = p_initial

    lines = []
    for i in eachindex(poly.exterior[1:(end - 1)])
        p_cur = gmsh.model.geo.addPoint(poly.exterior[i][2]..., 0, mesh_size)
        push!(lines, gmsh.model.geo.addLine(p_last, p_cur))
        p_last = p_cur
    end
    push!(lines, gmsh.model.geo.addLine(p_last, p_initial))
    return gmsh.model.geo.addCurveLoop(reverse(lines))
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
        gmsh.write(save_path)
    end
    gmsh.finalize()

    return Ferrite.Grid(elements, nodes, facetsets = facetsets, cellsets = cellsets)
end


"""
    get_convex_hull_and_buffer(mp::GeometryBasics.MultiPoint, buffer_width::Real)

Get convex hull and its buffer for mesh generation.

Returns (convex_hull_geom, buffered_geom) in GeometryBasics format.
"""
function get_convex_hull_and_buffer(mp::GeometryBasics.MultiPoint, buffer_width::Real)
    ch = LibGEOS.convexhull(mp)
    ch_gb = GeoInterface.convert(GeometryBasics, ch)

    outer_boundary = buffer(ch, buffer_width)
    outer_boundary_gb = GeoInterface.convert(GeometryBasics, outer_boundary)

    return ch_gb, outer_boundary_gb
end

function generate_mesh(
        mp::GeometryBasics.MultiPoint{1};
        element_order::Int = 1,
        save_path = nothing,
    )
    cloud_points = [tuple(p...) for p in mp.points]
    sizeMin, sizeMax, distMin, distMax, interior_mesh_size = auto_size_params(cloud_points)
    buffer_width = 1.5 * sizeMax

    coords = sort!([p[1] for p in mp.points])
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

    ch_points_inds = indexin([min_x, max_x], mp)
    mp_inner = mp.points[1:end .∉ Ref(ch_points_inds)]
    ps_inner = [gmsh.model.geo.addPoint(p..., 0, interior_mesh_size) for p in mp_inner]

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, ps_inner, 1, l_int)

    _create_physical_groups(1, [l_ext, l_int], [l_int])
    _configure_mesh_fields(ps_inner, sizeMin, sizeMax, distMin, distMax)

    gmsh.model.mesh.generate(1)
    return _finalize_mesh(element_order, save_path, 1)
end

function _create_physical_groups(dim, domain_entities, interior_entities)
    group = gmsh.model.addPhysicalGroup(dim, domain_entities)
    gmsh.model.setPhysicalName(dim, group, "Domain")
    int_group = gmsh.model.addPhysicalGroup(dim, interior_entities)
    gmsh.model.setPhysicalName(dim, int_group, "Interior")
    return group, int_group
end

@doc raw"""
    generate_mesh(mp::GeometryBasics.MultiPoint, buffer_width::Real,
                  interior_mesh_size::Real;
                  exterior_mesh_size::Real = 2 * interior_mesh_size,
                  element_order::Int = 1, save_path=nothing)

Generate a mesh for a spatial point cloud, with a buffer to counteract boundary
effects from the SPDE discretization.

### Input

- `mp` -- MultiPoint object
- `buffer_width` -- Width of the buffer around the convex hull
- `interior_mesh_size` -- Mesh size inside the convex hull
- `exterior_mesh_size` -- Mesh size outside the convex hull
- `element_order` -- Order of the element basis functions
- `save_path` -- Optional path to save the mesh

### Output

A `Ferrite.Grid` object

### Algorithm

1. Create the convex hull of the point cloud via LibGEOS
2. Create the buffer around the convex hull
3. Create a mesh for the buffered polygon using Gmsh
4. Transfer the Gmsh information to Ferrite
"""
function generate_mesh(
        mp::GeometryBasics.MultiPoint;
        element_order::Int = 1,
        save_path = nothing,
    )
    cloud_points = [tuple(p...) for p in mp.points]
    sizeMin, sizeMax, distMin, distMax, interior_mesh_size = auto_size_params(cloud_points)
    buffer_width = 1.5 * sizeMax

    ch_gb, outer_boundary_gb = get_convex_hull_and_buffer(mp, buffer_width)

    _setup_gmsh_common("Scattered mesh")

    c_ext = _add_poly_to_gmsh(outer_boundary_gb, sizeMax)
    c_int = _add_poly_to_gmsh(ch_gb, interior_mesh_size)
    s_int = gmsh.model.geo.addPlaneSurface([c_int])
    s_ext = gmsh.model.geo.addPlaneSurface([c_ext, c_int])

    ch_points = [l[1] for l in ch_gb.exterior]
    ch_points_inds = indexin(ch_points, mp)
    mp_inner = mp.points[1:end .∉ Ref(ch_points_inds)]
    ps_inner = [gmsh.model.geo.addPoint(p..., 0, interior_mesh_size) for p in mp_inner]

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, ps_inner, 2, s_int)

    _create_physical_groups(2, [s_ext, s_int], [s_int])
    _configure_mesh_fields(ps_inner, sizeMin, sizeMax, distMin, distMax)

    gmsh.model.mesh.generate(2)
    return _finalize_mesh(element_order, save_path, 2)
end

@doc raw"""
    generate_mesh(points; element_order::Int=1, save_path=nothing)

Generate a mesh from a list of points using Gmsh.

### Input

- `points` -- List of points (automatically converted to MultiPoint)
- `element_order` -- Order of the elements
- `save_path` -- Path to save the mesh

### Output

A Ferrite.Grid object
"""
function generate_mesh(
        points::AbstractVector;
        element_order::Int = 1,
        save_path = nothing,
    )
    mp = GeometryBasics.MultiPoint([GeometryBasics.Point(point...) for point in points])
    return generate_mesh(mp; element_order = element_order, save_path = save_path)
end
