using Gmsh, FerriteGmsh, Ferrite
using LibGEOS, GeometryBasics, GeoInterface

export generate_mesh

function _add_poly_to_gmsh(poly, mesh_size)
    p_initial = gmsh.model.geo.addPoint(poly.exterior[1][1]..., 0, mesh_size)
    p_last = p_initial

    lines = []
    for i in eachindex(poly.exterior[1:(end-1)])
        cur_line = poly.exterior[i]
        p_cur = gmsh.model.geo.addPoint(poly.exterior[i][2]..., 0, mesh_size)
        push!(lines, gmsh.model.geo.addLine(p_last, p_cur))
        p_last = p_cur
    end
    push!(lines, gmsh.model.geo.addLine(p_last, p_initial))
    return gmsh.model.geo.addCurveLoop(reverse(lines))
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
function generate_mesh(mp::GeometryBasics.MultiPoint, buffer_width::Real,
                       interior_mesh_size::Real;
                       exterior_mesh_size::Real = 2 * interior_mesh_size,
                       element_order::Int=1, save_path=nothing,)
    ch = LibGEOS.convexhull(mp)
    ch_gb = GeoInterface.convert(GeometryBasics, ch)
    outer_boundary = buffer(ch, buffer_width)
    outer_boundary_gb = GeoInterface.convert(GeometryBasics, outer_boundary)

    Gmsh.initialize()

    gmsh.model.add("Scattered mesh")

    c_ext = _add_poly_to_gmsh(outer_boundary_gb, exterior_mesh_size)
    c_int = _add_poly_to_gmsh(ch_gb, interior_mesh_size)
    s_int = gmsh.model.geo.addPlaneSurface([c_int])
    s_ext = gmsh.model.geo.addPlaneSurface([c_ext, c_int])

    ch_points = [l[1] for l in ch_gb.exterior]
    ch_points_inds = indexin(ch_points, mp)
    ps_inner = []
    for (i, inner_point) in enumerate(mp.points)
        if i ∉ ch_points_inds
            push!(ps_inner, gmsh.model.geo.addPoint(inner_point[1], inner_point[2], 0, interior_mesh_size))
        end
    end

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(0, ps_inner, 2, s_int)

    group = gmsh.model.addPhysicalGroup(2, [s_ext, s_int])
    gmsh.model.setPhysicalName(2, group, "Domain")
    int_group = gmsh.model.addPhysicalGroup(2, [s_int])
    gmsh.model.setPhysicalName(2, int_group, "Interior")
    #boundary_group = gmsh.model.addPhysicalGroup(1, [l_int_1, l_int_2, l_int_3, l_int_4])
    #gmsh.model.setPhysicalName(1, boundary_group, "Interior boundary")

    #ext_boundary_group =
        #gmsh.model.addPhysicalGroup(1, [l_ext_1, l_ext_2, l_ext_3, l_ext_4])
    #gmsh.model.setPhysicalName(1, ext_boundary_group, "Exterior boundary")

    gmsh.model.mesh.field.add("Constant", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", interior_mesh_size)
    gmsh.model.mesh.field.setNumber(1, "VOut", exterior_mesh_size)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [s_int])
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(2)
    if element_order != 1
        gmsh.model.mesh.setOrder(element_order)
    end
    gmsh.model.mesh.reverse()

    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Transfer the gmsh information
    nodes = FerriteGmsh.tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    domaincellset = cellsets["Domain"]
    elements = elements[collect(domaincellset)]

    boundarydict = toboundary(facedim)
    facesets = tofacetsets(boundarydict, elements)

    if save_path !== nothing
        gmsh.write(save_path)
    end
    gmsh.finalize()

    return Ferrite.Grid(elements, nodes, facesets = facesets, cellsets = cellsets)
end

@doc raw"""
    generate_mesh(points, buffer_width::Real, interior_mesh_size::Real;
                  exterior_mesh_size::Real=2 * interior_mesh_size,
                  element_order::Int=1, save_path=nothing)

Generate a mesh from a list of points using Gmsh.

### Input

- `points` -- List of points
- `buffer_width` -- Width of the buffer around the convex hull
- `interior_mesh_size` -- Mesh size inside the convex hull
- `exterior_mesh_size` -- Mesh size outside the convex hull
- `element_order` -- Order of the elements
- `save_path` -- Path to save the mesh

### Output

A Ferrite.Grid object
"""
function generate_mesh(points, buffer_width::Real, interior_mesh_size::Real;
                       exterior_mesh_size::Real = 2 * interior_mesh_size,
                       element_order::Int=1, save_path=nothing,)
    mp = GeometryBasics.MultiPoint([
        GeometryBasics.Point(cur_x, cur_y) for (cur_x, cur_y) in points
    ])
    return generate_mesh(mp, buffer_width, interior_mesh_size;
                         exterior_mesh_size=exterior_mesh_size,
                         element_order=element_order, save_path=save_path)
end
