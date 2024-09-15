using GMRFs, ForwardDiff

@testset "AD Jacobian" begin
    function f(x)
        return [x[1] + x[2], x[1] * x[2], cos(x[2])]
    end

    x₀ = [1.0, 2.0]

    full_jac = ForwardDiff.jacobian(f, x₀)
    J = ADJacobianMap(f, x₀, 3)

    @test size(J) == (3, 2)
    @test size(J') == (2, 3)

    v = rand(2)
    @test full_jac * v ≈ J * v

    v₂ = rand(3)
    @test full_jac' * v₂ ≈ J' * v₂
end
