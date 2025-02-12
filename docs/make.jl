using Documenter, GMRFs

include("generate_literate.jl")

makedocs(
    sitename = "GMRFs.jl",
    pages = Any[
        "Home"=>"index.md",
        "Tutorials"=>[
            "Overview" => "tutorials/index.md",
            "tutorials/autoregressive_models.md",
            "tutorials/spatial_modelling_spdes.md",
        ],
        "Autoregressive Models"=>"topics/autoregressive.md",
        "Spatial Models"=>"topics/spatial.md",
    ],
)
