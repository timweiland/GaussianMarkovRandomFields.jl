using GaussianMarkovRandomFields

@testset "Extension error hints" begin
    # A fresh process, so that no package extension is active
    script = """
    using GaussianMarkovRandomFields
    calls = (
        () -> evaluation_matrix(1, 2),
        () -> MaternModel([0.0 0.0; 1.0 1.0]; smoothness = 1),
        () -> MaternSPDE{2}(κ = 1.0, smoothness = 1),
        () -> contiguity_adjacency(1),
        () -> GaussianMarkovRandomFields.known_pattern_jacobian_backend(identity, [1.0]),
    )
    for call in calls
        try
            call()
            println("no error")
        catch err
            println(nameof(typeof(err)), " | ", replace(sprint(showerror, err), '\\n' => ' '))
        end
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, String)
    lines = split(chomp(out), '\n')
    @test length(lines) == 5
    @test all(startswith("MethodError |"), lines)
    fem = "Load Ferrite, FerriteGmsh, Gmsh, LibGEOS to activate it"
    @test occursin("`evaluation_matrix` is implemented in a package extension", lines[1]) && occursin(fem, lines[1])
    @test occursin("`MaternModel` is implemented in a package extension", lines[2]) && occursin(fem, lines[2])
    @test occursin("`MaternSPDE` is implemented in a package extension", lines[3]) && occursin(fem, lines[3])
    @test occursin("Load LibGEOS (and Shapefile", lines[4])
    @test occursin("Load SparseConnectivityTracer, SparseMatrixColorings", lines[5])

    # With the extension loaded, a wrong call gets a plain MethodError and no hint
    err = try
        evaluation_matrix("not a discretization")
    catch e
        e
    end
    @test err isa MethodError
    @test !occursin("package extension", sprint(showerror, err))
end
