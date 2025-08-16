using Gmsh, FerriteGmsh, Ferrite

export create_inflated_rectangle

"""
    create_inflated_rectangle(x0, y0, dx, dy, boundary_width, interior_mesh_size,
    exterior_mesh_size = 2 * interior_mesh_size; element_order = 1)

Create a triangular FEM discretization of a rectangle with an inflated boundary.
Useful for FEM discretizations of SPDEs, where the domain is often artificially
inflated to avoid undesirable boundary effects.
Mesh has physical groups "Domain", "Interior", "Interior boundary" and possibly
"Exterior boundary".

# Arguments
- `x0::Real`: x-coordinate of the bottom-left corner of the rectangle
- `y0::Real`: y-coordinate of the bottom-left corner of the rectangle
- `dx::Real`: Width of the rectangle
- `dy::Real`: Height of the rectangle
- `boundary_width::Real`: Width of the inflated boundary. If 0.0, mesh will not
                          be inflated at all.
- `interior_mesh_size::Real`: Mesh size in the interior of the rectangle
- `exterior_mesh_size::Real`: Mesh size in the exterior of the rectangle
- `element_order::Int`: Order of the FEM elements

# Returns
- `grid::Ferrite.Grid`: the FEM discretization of the rectangle
- `boundary_tags::Vector{Int}`: the indices of the boundary nodes

"""
function create_inflated_rectangle(
    x0,
    y0,
    dx,
    dy,
    boundary_width,
    interior_mesh_size,
    exterior_mesh_size = 2 * interior_mesh_size;
    element_order = 1,
)
    if boundary_width < 0.0
        throw(ArgumentError("boundary_width must be non-negative"))
    end
    Gmsh.initialize()

    # Disable verbose output
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add("t1")

    p_int_1 = gmsh.model.geo.addPoint(x0, y0, 0, interior_mesh_size)
    p_int_2 = gmsh.model.geo.addPoint(x0 + dx, y0, 0, interior_mesh_size)
    p_int_3 = gmsh.model.geo.addPoint(x0 + dx, y0 + dy, 0, interior_mesh_size)
    p_int_4 = gmsh.model.geo.addPoint(x0, y0 + dy, 0, interior_mesh_size)

    l_int_1 = gmsh.model.geo.addLine(p_int_1, p_int_2)
    l_int_2 = gmsh.model.geo.addLine(p_int_2, p_int_3)
    l_int_3 = gmsh.model.geo.addLine(p_int_3, p_int_4)
    l_int_4 = gmsh.model.geo.addLine(p_int_4, p_int_1)

    c_int = gmsh.model.geo.addCurveLoop([l_int_1, l_int_2, l_int_3, l_int_4])
    s_int = gmsh.model.geo.addPlaneSurface([c_int])
    gmsh.model.geo.mesh.setTransfiniteSurface(s_int, "Left")

    domain_group = [s_int]

    if boundary_width > 0.0
        p_ext_1 = gmsh.model.geo.addPoint(
            x0 - boundary_width,
            y0 - boundary_width,
            0,
            exterior_mesh_size,
        )
        p_ext_2 = gmsh.model.geo.addPoint(
            x0 + dx + boundary_width,
            y0 - boundary_width,
            0,
            exterior_mesh_size,
        )
        p_ext_3 = gmsh.model.geo.addPoint(
            x0 + dx + boundary_width,
            y0 + dy + boundary_width,
            0,
            exterior_mesh_size,
        )
        p_ext_4 = gmsh.model.geo.addPoint(
            x0 - boundary_width,
            y0 + dy + boundary_width,
            0,
            exterior_mesh_size,
        )

        l_ext_1 = gmsh.model.geo.addLine(p_ext_1, p_ext_2)
        l_ext_2 = gmsh.model.geo.addLine(p_ext_2, p_ext_3)
        l_ext_3 = gmsh.model.geo.addLine(p_ext_3, p_ext_4)
        l_ext_4 = gmsh.model.geo.addLine(p_ext_4, p_ext_1)

        c_ext = gmsh.model.geo.addCurveLoop([l_ext_1, l_ext_2, l_ext_3, l_ext_4])
        s_ext = gmsh.model.geo.addPlaneSurface([c_ext, c_int])

        domain_group = [s_ext, s_int]
    end

    gmsh.model.geo.synchronize()

    group = gmsh.model.addPhysicalGroup(2, domain_group)
    gmsh.model.setPhysicalName(2, group, "Domain")
    int_group = gmsh.model.addPhysicalGroup(2, [s_int])
    gmsh.model.setPhysicalName(2, int_group, "Interior")
    boundary_group = gmsh.model.addPhysicalGroup(1, [l_int_1, l_int_2, l_int_3, l_int_4])
    gmsh.model.setPhysicalName(1, boundary_group, "Interior boundary")

    if boundary_width > 0.0
        ext_boundary_group =
            gmsh.model.addPhysicalGroup(1, [l_ext_1, l_ext_2, l_ext_3, l_ext_4])
        gmsh.model.setPhysicalName(1, ext_boundary_group, "Exterior boundary")
    end

    gmsh.model.mesh.generate(2)
    if element_order != 1
        gmsh.model.mesh.setOrder(element_order)
    end

    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Transfer the gmsh information
    nodes = FerriteGmsh.tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    # "Domain" is the name of a PhysicalGroup and saves all cells that define the computational domain
    domaincellset = cellsets["Domain"]
    elements = elements[collect(domaincellset)]

    boundarydict = toboundary(facedim)
    facesets = tofacetsets(boundarydict, elements)

    gmsh.finalize()

    return Ferrite.Grid(elements, nodes, facesets = facesets, cellsets = cellsets)
end
