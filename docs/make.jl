using Documenter, DocumenterVitepress, GaussianMarkovRandomFields, DocumenterCitations
using Ferrite

include("generate_literate.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "GMRFs.jl",
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "tutorials/autoregressive_models.md",
            "tutorials/spatial_modelling_spdes.md",
            "tutorials/spatiotemporal_modelling.md",
            "tutorials/boundary_conditions.md",
            "tutorials/bernoulli_spatial_classification.md",
            "tutorials/bym_scotland_lip_cancer.md",
            "tutorials/automatic_differentiation.md",
        ],
        "API Reference" => [
            "Overview" => "reference/index.md",
            "GMRFs" => "reference/gmrfs.md",
            "Latent Models" => "reference/latent_models.md",
            "Observation Models" => "reference/observation_models.md",
            "Gaussian Approximation" => "reference/gaussian_approximation.md",
            "Hard Constraints" => "reference/hard_constraints.md",
            "Automatic Differentiation" => "reference/autodiff.md",
            "SPDEs" => "reference/spdes.md",
            "Discretizations" => "reference/discretizations.md",
            "Meshes" => "reference/meshes.md",
            "Plotting" => "reference/plotting.md",
            "Solvers" => "reference/solvers.md",
            "Autoregressive Models" => "reference/autoregressive.md",
            "Linear maps" => "reference/linear_maps.md",
            "Preconditioners" => "reference/preconditioners.md",
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
    modules = [GaussianMarkovRandomFields],
    checkdocs = :exports,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/timweiland/GaussianMarkovRandomFields.jl.git",
    devbranch = "main",
    push_preview = true
)
