using Documenter, DocumenterVitepress, GaussianMarkovRandomFields, DocumenterCitations
# Load the FEM weakdeps so the GaussianMarkovRandomFieldsFEM extension activates;
# its method definitions carry the docstrings cross-referenced from the API and
# developer docs.
using Ferrite, FerriteGmsh, Gmsh, LibGEOS

const GaussianMarkovRandomFieldsFEM =
    Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsFEM)
const GaussianMarkovRandomFieldsLibGEOS =
    Base.get_extension(GaussianMarkovRandomFields, :GaussianMarkovRandomFieldsLibGEOS)

include("generate_literate.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "GMRFs.jl",
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "tutorials/getting_started.md",
            "Building models" => [
                "tutorials/autoregressive_models.md",
                "tutorials/spatial_modelling_spdes.md",
                "tutorials/spatiotemporal_modelling.md",
                "tutorials/boundary_conditions.md",
            ],
            "Non-Gaussian observations" => [
                "tutorials/bernoulli_spatial_classification.md",
                "tutorials/bym_scotland_lip_cancer.md",
            ],
            "Hyperparameter inference" => [
                "tutorials/automatic_differentiation.md",
                "tutorials/autodiff_mcmc.md",
            ],
            "Further topics" => [
                "tutorials/workspace_factorization_reuse.md",
                "tutorials/kl_approximation.md",
                "tutorials/graphical_lasso.md",
            ],
        ],
        "API Reference" => [
            "Overview" => "reference/index.md",
            "Core types" => [
                "GMRFs" => "reference/gmrfs.md",
                "Hard Constraints" => "reference/hard_constraints.md",
            ],
            "Building models" => [
                "Latent Models" => "reference/latent_models.md",
                "Autoregressive Models" => "reference/autoregressive.md",
                "Formula Interface" => "reference/formula.md",
                "Spatial Utilities" => "reference/spatial_utils.md",
            ],
            "SPDE discretizations" => [
                "SPDEs" => "reference/spdes.md",
                "Discretizations" => "reference/discretizations.md",
                "Meshes" => "reference/meshes.md",
            ],
            "Observations and inference" => [
                "Observation Models" => "reference/observation_models.md",
                "Gaussian Approximation" => "reference/gaussian_approximation.md",
                "Automatic Differentiation" => "reference/autodiff.md",
            ],
            "Computation and performance" => [
                "Solvers" => "reference/solvers.md",
                "Preconditioners" => "reference/preconditioners.md",
                "Workspaces" => "reference/workspaces.md",
                "Linear maps" => "reference/linear_maps.md",
            ],
            "Further topics" => [
                "KL Approximations" => "reference/kl_approximation.md",
                "Graphical Lasso" => "reference/graphical_lasso.md",
                "Plotting" => "reference/plotting.md",
            ],
        ],
        "Bibliography" => "bibliography.md",
        "Developer Documentation" => [
            "Overview" => "dev-docs/index.md",
            "Solvers" => "dev-docs/solvers.md",
            "SPDEs" => "dev-docs/spdes.md",
            "Discretizations" => "dev-docs/discretizations.md",
        ],
    ],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/timweiland/GaussianMarkovRandomFields.jl",
        devbranch = "main",
        devurl = "dev"
    ),
    plugins = [bib],
    modules = [
        GaussianMarkovRandomFields,
        GaussianMarkovRandomFieldsFEM,
        GaussianMarkovRandomFieldsLibGEOS,
    ],
    checkdocs = :exports,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/timweiland/GaussianMarkovRandomFields.jl.git",
    devbranch = "main",
    push_preview = true
)
