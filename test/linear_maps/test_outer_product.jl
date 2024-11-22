using GMRFs, LinearMaps, LinearAlgebra, Random

@testset "Outer product map" begin
    rng = MersenneTwister(52398)
    A = LinearMap(rand(rng, 3, 4))
    Q = LinearMap(rand(rng, 3, 3))
    N = size(A, 2)
    x = rand(rng, N)

    opm = OuterProductMap(A, Q)
    @test size(opm) == (N, N)
    @test !issymmetric(opm)
    @test opm' isa OuterProductMap
    @test opm'.A === opm.A
    @test opm'.Q === opm.Q'
    @test Array(to_matrix(opm)) ≈ Array(to_matrix(A') * to_matrix(Q) * to_matrix(A))
    @test opm * x ≈ A' * Q * A * x

    Q_sym_sqrt = rand(rng, 3, 3)
    M = size(Q_sym_sqrt, 1)
    Q_sym = LinearMap(Q_sym_sqrt * Q_sym_sqrt')
    Q_sym_sqrt = LinearMap(Q_sym_sqrt)
    Q_sym = LinearMapWithSqrt(Q_sym, Q_sym_sqrt)

    opm_sym = OuterProductMap(A, Q_sym)
    @test issymmetric(opm_sym)
    opm_sqrt = linmap_sqrt(opm_sym)
    @test size(opm_sqrt) == (N, M)
end
