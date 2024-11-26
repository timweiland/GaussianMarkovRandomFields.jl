using Distributions, Ferrite, Random, SparseArrays, LinearAlgebra, Preconditioners

struct NonlinearOptimProblem
    f_and_Jf::Function
    solution::Function
end

example_problem = NonlinearOptimProblem(
    (x, A, u) -> ((A * u).^3 + x.^2 .- 1, Diagonal(3 * (A * u).^2) * A),
    x -> (1 .- x.^2).^(1 / 3)
)

@testset "Gauss-Newton Optimizer" begin
    rng = MersenneTwister(6692340)

    grid = generate_grid(Line, (200,))
    ip = Lagrange{RefLine, 1}()
    qr = QuadratureRule{RefLine}(2)
    disc = FEMDiscretization(grid, ip, qr)
    spde = MaternSPDE{1}(range=0.3, smoothness=1)
    x_prior = discretize(spde, disc)

    xs = -0.99:0.025:0.99
    A = evaluation_matrix(disc, [Tensors.Vec(x) for x in xs])

    common_params = (
        rand(rng, length(mean(x_prior))),
        precision_map(x_prior),
        u -> example_problem.f_and_Jf(sparse(xs), A, u),
        1e7, # Note: Optim result is very dependent on noise without line search
        spzeros(length(xs)),
    )
    solvers = [
        GNCholeskySolverBlueprint(),
        GNCGSolverBlueprint(),
        GNCGSolverBlueprint(
            preconditioner_fn = A -> Preconditioners.CholeskyPreconditioner(A)
        ),
    ]
    line_searches = [
        NoLineSearch(),
        BacktrackingLineSearch(),
    ]
    gt = example_problem.solution(xs)
    for solver_bp in solvers
        for line_search in line_searches
            gno = GaussNewtonOptimizer(
                common_params...,
                solver_bp=solver_bp,
                line_search=line_search,
            )
            optimize(gno)
            x_final = gno.xₖ
            rel_err = norm(A * x_final - gt) / norm(gt)
            @test rel_err < 0.01
        end
    end
end
