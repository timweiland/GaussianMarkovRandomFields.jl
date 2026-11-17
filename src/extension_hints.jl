################################################################################
#
#   Error hints for functionality that lives in package extensions.
#
#   These functions and types are declared in the main package but only gain
#   methods when their extension is loaded. Calling one without the extension
#   raises an ordinary `MethodError`; the hint below adds which packages to load.
#
################################################################################

const _FEM_EXTENSION = ((:GaussianMarkovRandomFieldsFEM,), "Ferrite, FerriteGmsh, Gmsh, LibGEOS")
const _SPARSE_AD_EXTENSION = (
    (:GaussianMarkovRandomFieldsSparseADLikelihoods,), "SparseConnectivityTracer, SparseMatrixColorings",
)
const _ADJACENCY_EXTENSIONS = (
    (:GaussianMarkovRandomFieldsLibGEOS, :GaussianMarkovRandomFieldsShapefile),
    "LibGEOS (and Shapefile to read adjacency from a shapefile)",
)

# Maps each function or type (without type parameters) to the extensions it needs
# and the packages that activate them.
const _EXTENSION_REQUIREMENTS = Dict{Any, Tuple{Tuple{Vararg{Symbol}}, String}}(
    # FEM functions
    evaluation_matrix => _FEM_EXTENSION,
    node_selection_matrix => _FEM_EXTENSION,
    derivative_matrices => _FEM_EXTENSION,
    second_derivative_matrices => _FEM_EXTENSION,
    assemble_mass_matrix => _FEM_EXTENSION,
    assemble_diffusion_matrix => _FEM_EXTENSION,
    assemble_advection_matrix => _FEM_EXTENSION,
    assemble_streamline_diffusion_matrix => _FEM_EXTENSION,
    lump_matrix => _FEM_EXTENSION,
    apply_soft_constraints! => _FEM_EXTENSION,
    assemble_C_G_matrices => _FEM_EXTENSION,
    product_matern => _FEM_EXTENSION,
    range_to_κ => _FEM_EXTENSION,
    smoothness_to_ν => _FEM_EXTENSION,
    matern_precision_only => _FEM_EXTENSION,
    barrier_triangles => _FEM_EXTENSION,
    joint_ssm => _FEM_EXTENSION,
    kronecker_product_spatiotemporal_model => _FEM_EXTENSION,
    create_inflated_rectangle => _FEM_EXTENSION,
    generate_mesh => _FEM_EXTENSION,
    PointEvaluationObsModel => _FEM_EXTENSION,
    PointDerivativeObsModel => _FEM_EXTENSION,
    PointSecondDerivativeObsModel => _FEM_EXTENSION,
    # FEM types whose constructors live in the extension
    FEMDiscretization => _FEM_EXTENSION,
    MaternSPDE => _FEM_EXTENSION,
    AdvectionDiffusionSPDE => _FEM_EXTENSION,
    ImplicitEulerSSM => _FEM_EXTENSION,
    ImplicitEulerJointSSMMatrices => _FEM_EXTENSION,
    ImplicitEulerConstantMeshSTGMRF => _FEM_EXTENSION,
    ConcreteConstantMeshSTGMRF => _FEM_EXTENSION,
    MaternModel => _FEM_EXTENSION,
    BarrierModel => _FEM_EXTENSION,
    # Polygon contiguity
    contiguity_adjacency => _ADJACENCY_EXTENSIONS,
    # Sparse AD backends
    default_sparse_jacobian_backend => _SPARSE_AD_EXTENSION,
    known_pattern_jacobian_backend => _SPARSE_AD_EXTENSION,
    known_pattern_hessian_backend => _SPARSE_AD_EXTENSION,
    residual_curvature => _SPARSE_AD_EXTENSION,
)

# A parametric constructor call such as `MaternSPDE{2}(...)` reports the concrete type.
_extension_key(f::Union{DataType, UnionAll}) = Base.typename(f).wrapper
_extension_key(f) = f

function _extension_hint(io::IO, exc::MethodError, argtypes, kwargs)
    requirement = get(_EXTENSION_REQUIREMENTS, _extension_key(exc.f), nothing)
    requirement === nothing && return nothing
    extensions, packages = requirement
    all(ext -> Base.get_extension(@__MODULE__, ext) !== nothing, extensions) && return nothing
    print(
        io, "\n\n`", nameof(_extension_key(exc.f)), "` is implemented in a package extension of ",
        "GaussianMarkovRandomFields that is not loaded. Load ", packages, " to activate it."
    )
    return nothing
end

function __init__()
    Base.Experimental.register_error_hint(_extension_hint, MethodError)
    return nothing
end
