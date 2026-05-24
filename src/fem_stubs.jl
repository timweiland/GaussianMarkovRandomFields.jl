################################################################################
#
#   Stubs for FEM-related functionality.
#
#   The implementations live in `ext/GaussianMarkovRandomFieldsFEM/`. The bare
#   type definitions live in `src/fem_types.jl` (so that other code in the main
#   package and other extensions can reference these types without loading the
#   FEM packages). The stubs declared below produce a helpful error if a user
#   tries to call a FEM helper without the extension active.
#
################################################################################

const _FEM_DEPS_HINT = """
Load the FEM dependencies first, e.g.:

    using Ferrite, FerriteGmsh, Gmsh, LibGEOS, GeometryBasics, Tensors, GeoInterface

This activates the `GaussianMarkovRandomFieldsFEM` package extension which
provides FEM discretizations, mesh generation, Matérn SPDEs and related
helpers.
"""

_fem_extension_required(name) = error(
    "`$name` requires the FEM extension to be active.\n" * _FEM_DEPS_HINT,
)

# --- FEM utility / evaluation helpers ---------------------------------------
function evaluation_matrix end
function node_selection_matrix end
function derivative_matrices end
function second_derivative_matrices end

evaluation_matrix(args...; kwargs...) = _fem_extension_required("evaluation_matrix")
node_selection_matrix(args...; kwargs...) = _fem_extension_required("node_selection_matrix")
derivative_matrices(args...; kwargs...) = _fem_extension_required("derivative_matrices")
second_derivative_matrices(args...; kwargs...) = _fem_extension_required("second_derivative_matrices")

export evaluation_matrix, node_selection_matrix,
    derivative_matrices, second_derivative_matrices

# --- FEM utility primitives --------------------------------------------------
function assemble_mass_matrix end
function assemble_diffusion_matrix end
function assemble_advection_matrix end
function assemble_streamline_diffusion_matrix end
function lump_matrix end
function apply_soft_constraints! end

assemble_mass_matrix(args...; kwargs...) = _fem_extension_required("assemble_mass_matrix")
assemble_diffusion_matrix(args...; kwargs...) = _fem_extension_required("assemble_diffusion_matrix")
assemble_advection_matrix(args...; kwargs...) = _fem_extension_required("assemble_advection_matrix")
assemble_streamline_diffusion_matrix(args...; kwargs...) =
    _fem_extension_required("assemble_streamline_diffusion_matrix")
lump_matrix(args...; kwargs...) = _fem_extension_required("lump_matrix")
apply_soft_constraints!(args...; kwargs...) =
    _fem_extension_required("apply_soft_constraints!")

export assemble_mass_matrix, assemble_diffusion_matrix, assemble_advection_matrix,
    assemble_streamline_diffusion_matrix, lump_matrix, apply_soft_constraints!

# --- SPDE helpers ------------------------------------------------------------
function assemble_C_G_matrices end
function product_matern end
function range_to_κ end
function smoothness_to_ν end
function matern_precision_only end

assemble_C_G_matrices(args...; kwargs...) = _fem_extension_required("assemble_C_G_matrices")
product_matern(args...; kwargs...) = _fem_extension_required("product_matern")
range_to_κ(args...; kwargs...) = _fem_extension_required("range_to_κ")
smoothness_to_ν(args...; kwargs...) = _fem_extension_required("smoothness_to_ν")
matern_precision_only(args...; kwargs...) = _fem_extension_required("matern_precision_only")

export assemble_C_G_matrices, product_matern, range_to_κ, smoothness_to_ν

# --- State-space model helpers (impls in FEM ext) ---------------------------
function joint_ssm end
function get_G end
function get_M end
function get_Σ⁻¹ end
function get_Σ⁻¹_sqrt end
function get_constraint_handler end
function get_constraint_noise end

joint_ssm(args...; kwargs...) = _fem_extension_required("joint_ssm")

export joint_ssm

# --- Spatiotemporal helpers --------------------------------------------------
function N_spatial end
function ssm end
function kronecker_product_spatiotemporal_model end

kronecker_product_spatiotemporal_model(args...; kwargs...) =
    _fem_extension_required("kronecker_product_spatiotemporal_model")

export kronecker_product_spatiotemporal_model

# --- Mesh helpers ------------------------------------------------------------
function create_inflated_rectangle end
function generate_mesh end

create_inflated_rectangle(args...; kwargs...) =
    _fem_extension_required("create_inflated_rectangle")
generate_mesh(args...; kwargs...) = _fem_extension_required("generate_mesh")

export create_inflated_rectangle, generate_mesh

# --- FEM observation models --------------------------------------------------
function PointEvaluationObsModel end
function PointDerivativeObsModel end
function PointSecondDerivativeObsModel end

PointEvaluationObsModel(args...; kwargs...) = _fem_extension_required("PointEvaluationObsModel")
PointDerivativeObsModel(args...; kwargs...) = _fem_extension_required("PointDerivativeObsModel")
PointSecondDerivativeObsModel(args...; kwargs...) =
    _fem_extension_required("PointSecondDerivativeObsModel")

export PointEvaluationObsModel, PointDerivativeObsModel, PointSecondDerivativeObsModel

# --- Adjacency from polygonal geometries (LibGEOS extension) -----------------
function contiguity_adjacency end

contiguity_adjacency(args...; kwargs...) = error(
    "`contiguity_adjacency` requires the LibGEOS extension to be active.\n" *
        "Load LibGEOS first:\n\n    using LibGEOS\n",
)

export contiguity_adjacency

# --- Internal hook for ConstantMeshSTGMRF preconditioner ---------------------
function default_preconditioner_strategy end
