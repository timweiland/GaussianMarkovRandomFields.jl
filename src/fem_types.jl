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

FEMDiscretization(args...; kwargs...) = _fem_extension_required("FEMDiscretization")
(::Type{<:FEMDiscretization})(args...; kwargs...) = _fem_extension_required("FEMDiscretization")

ndim(::FEMDiscretization{D}) where {D} = D

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

MaternSPDE(args...; kwargs...) = _fem_extension_required("MaternSPDE")
(::Type{<:MaternSPDE})(args...; kwargs...) = _fem_extension_required("MaternSPDE")

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

AdvectionDiffusionSPDE(args...; kwargs...) = _fem_extension_required("AdvectionDiffusionSPDE")
(::Type{<:AdvectionDiffusionSPDE})(args...; kwargs...) = _fem_extension_required("AdvectionDiffusionSPDE")

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

ImplicitEulerSSM(args...; kwargs...) = _fem_extension_required("ImplicitEulerSSM")

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

ImplicitEulerJointSSMMatrices(args...; kwargs...) =
    _fem_extension_required("ImplicitEulerJointSSMMatrices")

# --- Spatiotemporal constant-mesh metadata ----------------------------------
"""
    ImplicitEulerMetadata{D, SSM} <: GMRFMetadata

Metadata for implicit Euler spatiotemporal GMRFs with constant spatial mesh.
"""
struct ImplicitEulerMetadata{D, SSM} <: GMRFMetadata
    discretization::FEMDiscretization{D}
    ssm::SSM
    N_spatial::Int
    N_t::Int
end

"""
    ConcreteSTMetadata{D} <: GMRFMetadata

Metadata for concrete spatiotemporal GMRFs with constant spatial mesh.
"""
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

ImplicitEulerConstantMeshSTGMRF(args...; kwargs...) =
    _fem_extension_required("ImplicitEulerConstantMeshSTGMRF")
ConcreteConstantMeshSTGMRF(args...; kwargs...) =
    _fem_extension_required("ConcreteConstantMeshSTGMRF")

# --- Matérn latent model -----------------------------------------------------
"""
    MaternModel{F, S, Alg, C, P} <: LatentModel

A Matérn latent model for constructing spatial GMRFs from discretized Matérn
SPDEs. Validated user-facing constructors live in the
`GaussianMarkovRandomFieldsFEM` extension.
"""
struct MaternModel{F, S <: Integer, Alg, C, P} <: LatentModel
    discretization::F
    smoothness::S
    alg::Alg
    constraint::C
    observation_points::P
end

MaternModel(args...; kwargs...) = _fem_extension_required("MaternModel")
(::Type{<:MaternModel})(args...; kwargs...) = _fem_extension_required("MaternModel")
