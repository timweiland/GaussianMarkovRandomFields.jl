################################################################################
#
#   Type definitions for FEM-backed GMRF machinery.
#
#   Concrete construction logic and methods that actually use Ferrite / Gmsh
#   live in `ext/GaussianMarkovRandomFieldsFEM/`. Defining the types here lets
#   other extensions (and user code) reference them without first loading the
#   FEM weakdeps — the runtime errors come from the stub methods.
#
################################################################################

export FEMDiscretization, MaternSPDE, AdvectionDiffusionSPDE,
    MaternModel,
    ImplicitEulerSSM, ImplicitEulerJointSSMMatrices, JointSSMMatrices,
    ImplicitEulerMetadata, ConcreteSTMetadata,
    ImplicitEulerConstantMeshSTGMRF, ConcreteConstantMeshSTGMRF,
    ConstantMeshSTGMRF

# --- FEM discretization ------------------------------------------------------
"""
    FEMDiscretization{D, S, G, I, Q, GI, H, CH}

Holds the data needed to discretize an (S)PDE using a Finite Element Method.

The validated user-facing constructor lives in the `GaussianMarkovRandomFieldsFEM`
extension and requires Ferrite to be loaded. See its docstring for details.
"""
struct FEMDiscretization{D, S, G, I, Q, GI, H, CH}
    grid::G
    interpolation::I
    quadrature_rule::Q
    geom_interpolation::GI
    dof_handler::H
    constraint_handler::CH
    constraint_noise::Vector{Float64}
end

# COV_EXCL_START
FEMDiscretization(args...; kwargs...) = _fem_extension_required("FEMDiscretization")
(::Type{<:FEMDiscretization})(args...; kwargs...) = _fem_extension_required("FEMDiscretization")
# COV_EXCL_STOP

"""
    ndim(f::FEMDiscretization)

Return the dimension of space in which the discretization is defined.
Typically `ndim(f) == 1`, `2`, or `3`.
"""
ndim(::FEMDiscretization{D}) where {D} = D

# Generic `discretize` fallback over (::SPDE, ::FEMDiscretization). Lives here
# rather than next to the SPDE abstract type because it references
# `FEMDiscretization`, which is only defined after `spdes/spde.jl` runs.
discretize(s::SPDE, d::FEMDiscretization) = throw(MethodError(discretize, (s, d))) # COV_EXCL_LINE

# --- SPDEs -------------------------------------------------------------------
"""
    MaternSPDE{D, Tv, Ti}

The Whittle-Matérn SPDE. Validated user-facing constructors live in the
`GaussianMarkovRandomFieldsFEM` extension.
"""
struct MaternSPDE{D, Tv <: Real, Ti <: Integer} <: SPDE
    κ::Tv
    ν::Rational{Ti}
    σ²::Tv
    diffusion_factor::Matrix{Tv}
end

# COV_EXCL_START
MaternSPDE(args...; kwargs...) = _fem_extension_required("MaternSPDE")
(::Type{<:MaternSPDE})(args...; kwargs...) = _fem_extension_required("MaternSPDE")
# COV_EXCL_STOP

ndim(::MaternSPDE{D}) where {D} = D

"""
    AdvectionDiffusionSPDE{D}

Spatiotemporal advection-diffusion SPDE. Validated constructors live in the
`GaussianMarkovRandomFieldsFEM` extension.
"""
struct AdvectionDiffusionSPDE{D} <: SPDE
    κ::Real
    α::Rational
    H::AbstractMatrix
    γ::AbstractVector
    c::Real
    τ::Real
    spatial_spde::SPDE
    initial_spde::SPDE
end

# COV_EXCL_START
AdvectionDiffusionSPDE(args...; kwargs...) = _fem_extension_required("AdvectionDiffusionSPDE")
(::Type{<:AdvectionDiffusionSPDE})(args...; kwargs...) = _fem_extension_required("AdvectionDiffusionSPDE")
# COV_EXCL_STOP

# --- State-space models ------------------------------------------------------
"""
    JointSSMMatrices

Abstract type for the matrices defining the transition of a linear state-space
model. Concrete subtypes (and their construction logic) live in the
`GaussianMarkovRandomFieldsFEM` extension.
"""
abstract type JointSSMMatrices end

"""
    ImplicitEulerSSM

State-space model for the implicit Euler discretization of an SDE. The
user-facing constructor lives in the `GaussianMarkovRandomFieldsFEM` extension.
"""
struct ImplicitEulerSSM{X, S, GF, MF, MIF, BF, BIF, TS, C, V}
    x₀::X
    G::GF
    M::MF
    M⁻¹::MIF
    β::BF
    β⁻¹::BIF
    spatial_noise::S
    ts::TS
    constraint_handler::C
    constraint_noise::V
end

# COV_EXCL_START
ImplicitEulerSSM(args...; kwargs...) = _fem_extension_required("ImplicitEulerSSM")
# COV_EXCL_STOP

"""
    ImplicitEulerJointSSMMatrices

Joint SSM transition matrices for the implicit Euler scheme.
"""
struct ImplicitEulerJointSSMMatrices{T, GM, MM, SM, SQRT, C, V} <: JointSSMMatrices
    Δt::T
    G::GM
    M::MM
    Σ⁻¹::SM
    Σ⁻¹_sqrt::SQRT
    constraint_handler::C
    constraint_noise::V
end

# COV_EXCL_START
ImplicitEulerJointSSMMatrices(args...; kwargs...) =
    _fem_extension_required("ImplicitEulerJointSSMMatrices")
# COV_EXCL_STOP

# --- Spatiotemporal constant-mesh metadata ----------------------------------
# Internal metadata wrappers for the implicit-Euler and concrete spatiotemporal
# `MetaGMRF`s. Not part of the user-facing API.
struct ImplicitEulerMetadata{D, SSM} <: GMRFMetadata
    discretization::FEMDiscretization{D}
    ssm::SSM
    N_spatial::Int
    N_t::Int
end

struct ConcreteSTMetadata{D} <: GMRFMetadata
    discretization::FEMDiscretization{D}
    N_spatial::Int
    N_t::Int
end

"""
    ImplicitEulerConstantMeshSTGMRF{D, SSM, T, P, G}

Implicit Euler spatiotemporal GMRF with constant spatial discretization, as a
`MetaGMRF` over `ImplicitEulerMetadata`.
"""
const ImplicitEulerConstantMeshSTGMRF{D, SSM, T, P, G} =
    MetaGMRF{ImplicitEulerMetadata{D, SSM}, T, P, G}

"""
    ConcreteConstantMeshSTGMRF{D, T, P, G}

Concrete spatiotemporal GMRF with constant spatial discretization, as a
`MetaGMRF` over `ConcreteSTMetadata`.
"""
const ConcreteConstantMeshSTGMRF{D, T, P, G} =
    MetaGMRF{ConcreteSTMetadata{D}, T, P, G}

const ConstantMeshSTGMRF{D} =
    Union{ImplicitEulerConstantMeshSTGMRF{D}, ConcreteConstantMeshSTGMRF}

# COV_EXCL_START
ImplicitEulerConstantMeshSTGMRF(args...; kwargs...) =
    _fem_extension_required("ImplicitEulerConstantMeshSTGMRF")
ConcreteConstantMeshSTGMRF(args...; kwargs...) =
    _fem_extension_required("ConcreteConstantMeshSTGMRF")
# COV_EXCL_STOP

# --- Matérn latent model -----------------------------------------------------
"""
    MaternModel{F, S, Alg, C, P, M} <: LatentModel

A Matérn latent model for constructing spatial GMRFs from discretized Matérn
SPDEs. Validated user-facing constructors live in the
`GaussianMarkovRandomFieldsFEM` extension.

The κ-independent FEM matrices (the lumped mass matrix `C` and the stiffness
matrix `G`) are assembled once at construction and cached in `fem_matrices`, so
that repeated `precision_matrix` calls only redo the κ-dependent work.
"""
struct MaternModel{F, S <: Integer, Alg, C, P, M} <: LatentModel
    discretization::F
    smoothness::S
    alg::Alg
    constraint::C
    observation_points::P
    fem_matrices::M
end

# COV_EXCL_START
MaternModel(args...; kwargs...) = _fem_extension_required("MaternModel")
(::Type{<:MaternModel})(args...; kwargs...) = _fem_extension_required("MaternModel")
# COV_EXCL_STOP
