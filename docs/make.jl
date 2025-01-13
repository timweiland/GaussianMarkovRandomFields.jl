using Documenter, GMRFs

include("generate_literate.jl")

makedocs(
    sitename = "GMRFs.jl",
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Overview" => "tutorials/index.md",
            "tutorials/autoregressive_models.md",
        ],
        "Autoregressive Models" => "topics/autoregressive.md"
    ],
)
