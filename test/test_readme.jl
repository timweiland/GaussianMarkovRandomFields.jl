using GaussianMarkovRandomFields, Markdown

@testset "README snippets" begin
    readme = read(joinpath(pkgdir(GaussianMarkovRandomFields), "README.md"), String)
    blocks = [
        c.code for c in Markdown.parse(readme).content
            if c isa Markdown.Code && strip(c.language) == "julia"
    ]
    @test !isempty(blocks)

    # A fresh process, so the snippets must bring their own imports: the test suite has
    # already activated the package extensions in this one.
    script = join(blocks, "\n")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test success(run(ignorestatus(cmd)))
end
