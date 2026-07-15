module GaussianMarkovRandomFieldsFEM

using GaussianMarkovRandomFields
using GaussianMarkovRandomFields:
    AbstractGMRF, GMRF, MetaGMRF, GMRFMetadata, LatentModel,
    ExponentialFamily, LinearlyTransformedObservationModel,
    SPDE,
    AbstractSpatiotemporalGMRF,
    FEMDiscretization, MaternSPDE, AdvectionDiffusionSPDE,
    JointSSMMatrices, ImplicitEulerSSM, ImplicitEulerJointSSMMatrices,
    ImplicitEulerMetadata, ConcreteSTMetadata,
    ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF,
    ConstantMeshSTGMRF,
    MaternModel, BarrierModel,
    precision_map, to_matrix, make_chunks,
    SymmetricBlockTridiagonalMap, SSMBidiagonalMap, ZeroMap,
    temporal_block_gauss_seidel,
    _process_constraint, _ones_pattern

import GaussianMarkovRandomFields:
    ndim, evaluation_matrix, node_selection_matrix,
    derivative_matrices, second_derivative_matrices,
    assemble_mass_matrix, assemble_diffusion_matrix, assemble_advection_matrix,
    lump_matrix, assemble_streamline_diffusion_matrix, apply_soft_constraints!,
    discretize,
    assemble_C_G_matrices, product_matern,
    range_to_κ, smoothness_to_ν, matern_precision_only, barrier_triangles,
    joint_ssm,
    get_G, get_M, get_Σ⁻¹, get_Σ⁻¹_sqrt,
    get_constraint_handler, get_constraint_noise,
    N_t, N_spatial, time_means, time_vars, time_stds, time_rands,
    discretization_at_time, ssm,
    default_preconditioner_strategy,
    kronecker_product_spatiotemporal_model,
    create_inflated_rectangle, generate_mesh,
    PointEvaluationObsModel, PointDerivativeObsModel, PointSecondDerivativeObsModel,
    hyperparameters, model_name, precision_matrix, constraints, mean

using LinearAlgebra
using SparseArrays
using LinearMaps
using LinearSolve
using SpecialFunctions
using Statistics
using Random
using Distributions
using NearestNeighbors

using Ferrite
using Ferrite: Grid, Interpolation, QuadratureRule, DofHandler, ConstraintHandler,
    CellValues, CellIterator, CellCache, PointEvalHandler,
    Lagrange, RefLine, RefTriangle, Vec, ⊗,
    add!, close!, dof_range, celldofs, getcoordinates, getnbasefunctions,
    getnquadpoints, getdetJdV, shape_value, shape_gradient,
    reference_shape_value, reference_shape_gradient, gradient, hessian,
    start_assemble, assemble!, apply!, allocate_matrix,
    generate_grid
using FerriteGmsh
using FerriteGmsh: tonodes, toelements, tocellsets, toboundary, tofacetsets
using Gmsh
using Gmsh: gmsh
using LibGEOS

include("GaussianMarkovRandomFieldsFEM/fem_discretization.jl")
include("GaussianMarkovRandomFieldsFEM/fem_derivatives.jl")
include("GaussianMarkovRandomFieldsFEM/fem_utils.jl")
include("GaussianMarkovRandomFieldsFEM/linear_ssm.jl")
include("GaussianMarkovRandomFieldsFEM/implicit_euler_ssm.jl")
include("GaussianMarkovRandomFieldsFEM/constant_mesh.jl")
include("GaussianMarkovRandomFieldsFEM/matern_spde.jl")
include("GaussianMarkovRandomFieldsFEM/advection_diffusion.jl")
include("GaussianMarkovRandomFieldsFEM/product.jl")
include("GaussianMarkovRandomFieldsFEM/mesh_rectangle.jl")
include("GaussianMarkovRandomFieldsFEM/mesh_scattered.jl")
include("GaussianMarkovRandomFieldsFEM/matern_model.jl")
include("GaussianMarkovRandomFieldsFEM/barrier_model.jl")
include("GaussianMarkovRandomFieldsFEM/fem_obs_models.jl")

end
