using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, LinearSolve, Distributions

@testset "LinearSolve Architecture Tests" begin
    # Setup test GMRF - build Q from square root for RBMC testing
    n = 10
    L = spdiagm(0 => ones(n), -1 => -0.5*ones(n-1))  # Lower triangular
    Q_sqrt = L  
    Q = L * L'  # Build Q from square root
    μ = zeros(n)
    
    @testset "Capability Detection" begin
        # Test selinv and backward_solve capability detection
        test_cases = [
            ("CHOLMODFactorization", LinearSolve.CHOLMODFactorization(), Val{true}(), Val{true}()),
            ("CholeskyFactorization", LinearSolve.CholeskyFactorization(), Val{true}(), Val{true}()),
            ("KrylovJL_CG", LinearSolve.KrylovJL_CG(), Val{false}(), Val{false}()),
        ]
        
        for (name, alg, expected_selinv, expected_backward) in test_cases
            gmrf = GMRF(μ, Q, alg; Q_sqrt=Q_sqrt)
            actual_selinv = GaussianMarkovRandomFields.supports_selinv(gmrf.linsolve_cache.alg)
            actual_backward = GaussianMarkovRandomFields.supports_backward_solve(gmrf.linsolve_cache.alg)
            
            @test actual_selinv == expected_selinv
            @test actual_backward == expected_backward
        end
        
        # Test default algorithm supports both
        gmrf_default = GMRF(μ, Q; Q_sqrt=Q_sqrt)
        @test GaussianMarkovRandomFields.supports_selinv(gmrf_default.linsolve_cache.alg) == Val{true}()
        @test GaussianMarkovRandomFields.supports_backward_solve(gmrf_default.linsolve_cache.alg) == Val{true}()
    end
    
    @testset "RBMC Fallback with Q_sqrt" begin
        # Create GMRF with selinv-capable algorithm
        gmrf_selinv = GMRF(μ, Q, LinearSolve.CHOLMODFactorization(); Q_sqrt=Q_sqrt)
        @test GaussianMarkovRandomFields.supports_selinv(gmrf_selinv.linsolve_cache.alg) == Val{true}()
        
        # Create GMRF with Krylov method (should use RBMC)
        gmrf_rbmc = GMRF(μ, Q, LinearSolve.KrylovJL_CG(); Q_sqrt=Q_sqrt)
        @test GaussianMarkovRandomFields.supports_selinv(gmrf_rbmc.linsolve_cache.alg) == Val{false}()
        @test gmrf_rbmc.Q_sqrt !== nothing  # Should have Q_sqrt for sampling
        
        # Both should compute variance successfully
        var_selinv = var(gmrf_selinv)
        var_rbmc = var(gmrf_rbmc)
        
        @test length(var_selinv) == n
        @test length(var_rbmc) == n
        @test all(var_selinv .> 0)
        @test all(var_rbmc .> 0)
        
        # RBMC should be approximately equal to selinv (within RBMC sampling error)
        @test isapprox(var_selinv, var_rbmc, rtol=0.1)  # RBMC is stochastic
        
        # Test that RBMC actually uses the fallback strategy
        @test gmrf_rbmc.rbmc_strategy isa RBMCStrategy
    end
    
    @testset "RBMC vs Exact Variance" begin
        # For this simple structure, we can compute exact variance analytically
        Q_dense = Matrix(Q)
        var_exact = diag(inv(Q_dense))
        
        # Test selinv matches exact
        gmrf_exact = GMRF(μ, Q, LinearSolve.CHOLMODFactorization(); Q_sqrt=Q_sqrt)
        var_selinv = var(gmrf_exact)
        @test isapprox(var_selinv, var_exact, rtol=1e-10)
        
        # Test RBMC approximates exact
        gmrf_rbmc = GMRF(μ, Q, LinearSolve.KrylovJL_CG(); Q_sqrt=Q_sqrt)
        var_rbmc = var(gmrf_rbmc)
        @test isapprox(var_rbmc, var_exact, rtol=0.15)  # RBMC has sampling error
    end
    
    @testset "Algorithm Robustness" begin
        algorithms = [
            LinearSolve.CHOLMODFactorization(),
            LinearSolve.KrylovJL_CG(),  # Now works with Q_sqrt
        ]
        
        for alg in algorithms
            gmrf = GMRF(μ, Q, alg; Q_sqrt=Q_sqrt)
            
            # Basic operations should always work
            @test length(gmrf) == n
            @test mean(gmrf) == μ
            @test precision_matrix(gmrf) ≈ Q
            @test information_vector(gmrf) ≈ zeros(n)
            
            # Variance computation should work (selinv or RBMC)
            v = var(gmrf)
            @test length(v) == n
            @test all(v .> 0)
            
            # Sampling should work with Q_sqrt
            sample = rand(gmrf)
            @test length(sample) == n
        end
    end
end
