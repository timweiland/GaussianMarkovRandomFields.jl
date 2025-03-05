using Documenter, GMRFs, DocumenterCitations

include("generate_literate.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "GMRFs.jl",
    pages = Any[
        "Home"=>"index.md",
        "Tutorials"=>[
            "Overview" => "tutorials/index.md",
            "tutorials/autoregressive_models.md",
            "tutorials/spatial_modelling_spdes.md",
            "tutorials/boundary_conditions.md",
        ],
        "Autoregressive Models"=>"topics/autoregressive.md",
        "Spatial Models"=>"topics/spatial.md",
        "Plotting"=>"topics/plotting.md",
        "Solvers"=>"topics/solvers.md",
        "Linear maps"=>"topics/linear_maps.md",
        "References"=>"references.md",
    ],
    format = Documenter.HTML(
        assets=String["assets/citations.css"],
    ),
    plugins=[bib],
)
