using Documenter, GaussianMarkovRandomFields, DocumenterCitations
using Ferrite

include("generate_literate.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "GaussianMarkovRandomFields.jl",
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "tutorials/autoregressive_models.md",
            "tutorials/spatial_modelling_spdes.md",
            "tutorials/spatiotemporal_modelling.md",
            "tutorials/boundary_conditions.md",
        ],
        "API Reference" => [
            "Overview" => "reference/index.md",
            "GMRFs" => "reference/gmrfs.md",
            "Observation Models" => "reference/observation_models.md",
            "Gaussian Approximation" => "reference/gaussian_approximation.md",
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
    format = Documenter.HTML(assets = String["assets/citations.css"], collapselevel = 1),
    plugins = [bib],
    modules = [GaussianMarkovRandomFields],
    checkdocs = :exports,
)

deploydocs(repo = "github.com/timweiland/GaussianMarkovRandomFields.jl.git")
