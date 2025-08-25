using ForwardDiff
using StatsFuns
import GaussianMarkovRandomFields as GMRFs

@testset "Link Functions" begin

    @testset "IdentityLink" begin
        link = IdentityLink()
        x = [1.0, 2.0, -1.0]

        @test apply_link.(Ref(link), x) == x
        @test apply_invlink.(Ref(link), x) == x
        @test all(GMRFs.derivative_invlink.(Ref(link), x) .== 1.0)
        @test all(GMRFs.second_derivative_invlink.(Ref(link), x) .== 0.0)
    end

    @testset "LogLink" begin
        link = LogLink()
        x = [1.0, 2.0, 0.5]

        @test apply_link.(Ref(link), x) ≈ log.(x)
        @test apply_invlink.(Ref(link), log.(x)) ≈ x

        # Test derivatives
        for xi in x
            @test GMRFs.derivative_invlink(link, log(xi)) ≈ xi
            @test GMRFs.second_derivative_invlink(link, log(xi)) ≈ xi
        end

        # Compare with ForwardDiff
        for xi in x
            η = log(xi)
            @test GMRFs.derivative_invlink(link, η) ≈ ForwardDiff.derivative(x -> apply_invlink(link, x), η)
            @test GMRFs.second_derivative_invlink(link, η) ≈ ForwardDiff.derivative(x -> GMRFs.derivative_invlink(link, x), η)
        end
    end

    @testset "LogitLink" begin
        link = LogitLink()
        p = [0.1, 0.5, 0.9]

        @test apply_link.(Ref(link), p) ≈ logit.(p)
        @test apply_invlink.(Ref(link), logit.(p)) ≈ p rtol = 1.0e-10

        # Test derivatives
        for pi in p
            η = logit(pi)
            expected_deriv = pi * (1 - pi)
            @test GMRFs.derivative_invlink(link, η) ≈ expected_deriv rtol = 1.0e-10

            expected_second_deriv = pi * (1 - pi) * (1 - 2 * pi)
            @test GMRFs.second_derivative_invlink(link, η) ≈ expected_second_deriv rtol = 1.0e-10
        end

        # Compare with ForwardDiff
        for pi in p
            η = logit(pi)
            @test GMRFs.derivative_invlink(link, η) ≈ ForwardDiff.derivative(x -> apply_invlink(link, x), η) rtol = 1.0e-10
            @test GMRFs.second_derivative_invlink(link, η) ≈ ForwardDiff.derivative(x -> GMRFs.derivative_invlink(link, x), η) rtol = 1.0e-10
        end
    end

    @testset "Link Function Type Stability" begin
        @test @inferred apply_link(IdentityLink(), 1.0) == 1.0
        @test @inferred apply_invlink(LogLink(), 1.0) == exp(1.0)
        @test @inferred GMRFs.derivative_invlink(LogitLink(), 0.0) == 0.25
    end
end
