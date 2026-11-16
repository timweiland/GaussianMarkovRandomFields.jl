################################################################################
#
#   Stubs for FEM-related functionality.
#
#   The implementations live in `ext/GaussianMarkovRandomFieldsFEM/`. The bare
#   type definitions live in `src/fem/types.jl` (so that other code in the main
#   package and other extensions can reference these types without loading the
#   FEM packages). Calling one of these functions without the extension raises a
#   `MethodError` with a hint on what to load; see `src/extension_hints.jl`.
#
################################################################################

# --- FEM utility / evaluation helpers ---------------------------------------
function evaluation_matrix end
function node_selection_matrix end
function derivative_matrices end
function second_derivative_matrices end


export evaluation_matrix, node_selection_matrix,
    derivative_matrices, second_derivative_matrices

# --- FEM utility primitives --------------------------------------------------
function assemble_mass_matrix end
function assemble_diffusion_matrix end
function assemble_advection_matrix end
function assemble_streamline_diffusion_matrix end
function lump_matrix end
function apply_soft_constraints! end


export assemble_mass_matrix, assemble_diffusion_matrix, assemble_advection_matrix,
    assemble_streamline_diffusion_matrix, lump_matrix, apply_soft_constraints!

# --- SPDE helpers ------------------------------------------------------------
function assemble_C_G_matrices end
function product_matern end
function range_to_κ end
function smoothness_to_ν end
function matern_precision_only end
function barrier_triangles end


export assemble_C_G_matrices, product_matern, range_to_κ, smoothness_to_ν
export barrier_triangles

# --- State-space model helpers (impls in FEM ext) ---------------------------
function joint_ssm end
function get_G end
function get_M end
function get_Σ⁻¹ end
function get_Σ⁻¹_sqrt end
function get_constraint_handler end
function get_constraint_noise end


export joint_ssm

# --- Spatiotemporal helpers --------------------------------------------------
function N_spatial end
function ssm end
function kronecker_product_spatiotemporal_model end


export kronecker_product_spatiotemporal_model

# --- Mesh helpers ------------------------------------------------------------
function create_inflated_rectangle end
function generate_mesh end


export create_inflated_rectangle, generate_mesh

# --- FEM observation models --------------------------------------------------
function PointEvaluationObsModel end
function PointDerivativeObsModel end
function PointSecondDerivativeObsModel end


export PointEvaluationObsModel, PointDerivativeObsModel, PointSecondDerivativeObsModel

# --- Adjacency from polygonal geometries (LibGEOS extension) -----------------
function contiguity_adjacency end


export contiguity_adjacency

# --- Internal hook for ConstantMeshSTGMRF preconditioner ---------------------
function default_preconditioner_strategy end
