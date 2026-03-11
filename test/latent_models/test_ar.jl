using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using LinearSolve

# Build the n×n autocorrelation matrix Σ for a stationary AR process with
# coefficients phi, using the companion-form state-space representation
# and the discrete Lyapunov equation.
function _test_ar_autocorrelation_matrix(n, phi)
    p = length(phi)
    p == 0 && return Matrix(1.0I, n, n)

    # Companion matrix F and innovation covariance R
    F = zeros(p, p)
    F[1, :] .= phi
    for i in 2:p
        F[i, i - 1] = 1.0
    end
    R = zeros(p, p)
    R[1, 1] = 1.0

    # Solve discrete Lyapunov equation P = F P F' + R via vectorization
    vecP = (I(p^2) - kron(F, F)) \ vec(R)
    P_state = reshape(vecP, p, p)
    P_state ./= P_state[1, 1]  # normalize to unit marginal variance

    # Compute autocorrelations γ(k) = (F^k P)[1,1]
    gamma = zeros(n)
    gamma[1] = 1.0
    Fk = Matrix(1.0I, p, p)
    for k in 1:(n - 1)
        Fk = F * Fk
        gamma[k + 1] = (Fk * P_state)[1, 1]
    end

    # Toeplitz autocorrelation matrix
    return [gamma[abs(i - j) + 1] for i in 1:n, j in 1:n]
end

# Reference Durbin-Levinson: PACF → AR coefficients
function _test_pacf_to_ar(pacf)
    p = length(pacf)
    phi = zeros(p)
    phi[1] = pacf[1]
    for k in 2:p
        phi_prev = copy(phi)
        phi[k] = pacf[k]
        for j in 1:(k - 1)
            phi[j] = phi_prev[j] - pacf[k] * phi_prev[k - j]
        end
    end
    return phi
end

@testset "ARModel" begin
    # ============================================================
    # AR1 backward compatibility (AR1Model = ARModel{1})
    # ============================================================
    @testset "AR1 Backward Compatibility" begin
        @testset "Type alias" begin
            @test AR1Model === ARModel{1}
        end

        @testset "Constructor" begin
            model = AR1Model(5)
            @test model.n == 5
            @test model isa ARModel{1}

            @test_throws ArgumentError AR1Model(0)
            @test_throws ArgumentError AR1Model(-1)
        end

        @testset "Hyperparameters" begin
            model = AR1Model(3)
            params = hyperparameters(model)
            @test params == (τ = Real, ρ = Real)
        end

        @testset "Parameter Validation" begin
            model = AR1Model(3)
            @test precision_matrix(model; τ = 1.0, ρ = 0.5) isa SymTridiagonal
            @test precision_matrix(model; τ = 2.0, ρ = -0.9) isa SymTridiagonal
            @test precision_matrix(model; τ = 0.1, ρ = 0.0) isa SymTridiagonal

            @test_throws ArgumentError precision_matrix(model; τ = 0.0, ρ = 0.5)
            @test_throws ArgumentError precision_matrix(model; τ = -1.0, ρ = 0.5)
            @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = 1.0)
            @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = -1.0)
            @test_throws ArgumentError precision_matrix(model; τ = 1.0, ρ = 1.1)
        end

        @testset "Precision Matrix Structure" begin
            @testset "n=1 case" begin
                model = AR1Model(1)
                Q = precision_matrix(model; τ = 2.0, ρ = 0.5)
                @test Q isa SymTridiagonal
                @test size(Q) == (1, 1)
                @test Q[1, 1] == 2.0
            end

            @testset "n=2 case" begin
                model = AR1Model(2)
                τ, ρ = 2.0, 0.5
                Q = precision_matrix(model; τ = τ, ρ = ρ)
                @test Q isa SymTridiagonal
                @test size(Q) == (2, 2)
                @test Q[1, 1] == τ
                @test Q[2, 2] == τ
                @test Q[1, 2] == -ρ * τ
            end

            @testset "n=3 case" begin
                model = AR1Model(3)
                τ, ρ = 2.0, 0.5
                Q = precision_matrix(model; τ = τ, ρ = ρ)
                @test Q isa SymTridiagonal
                @test size(Q) == (3, 3)

                expected = [
                    τ           -ρ * τ         0;
                    -ρ * τ   (1 + ρ^2) * τ    -ρ * τ;
                    0          -ρ * τ         τ
                ]
                @test Matrix(Q) ≈ expected
            end

            @testset "n=5 general case" begin
                model = AR1Model(5)
                τ, ρ = 1.5, 0.7
                Q = precision_matrix(model; τ = τ, ρ = ρ)
                @test Q isa SymTridiagonal
                @test size(Q) == (5, 5)

                @test Q[1, 1] == τ
                @test Q[5, 5] == τ
                for i in 2:4
                    @test Q[i, i] == (1 + ρ^2) * τ
                end
                for i in 1:4
                    @test Q[i, i + 1] == -ρ * τ
                end
                @test Q[1, 3] == 0.0
            end
        end

        @testset "Mean Vector" begin
            for n in [1, 3, 10]
                model = AR1Model(n)
                μ = mean(model; τ = 1.0, ρ = 0.5)
                @test μ == zeros(n)
                @test length(μ) == n
            end
        end

        @testset "Constraints" begin
            model = AR1Model(5)
            @test constraints(model; τ = 1.0, ρ = 0.5) === nothing
        end

        @testset "GMRF Construction" begin
            model = AR1Model(4)
            τ, ρ = 1.2, 0.6
            gmrf = model(τ = τ, ρ = ρ)

            @test gmrf isa GMRF
            @test length(gmrf) == 4
            @test mean(gmrf) == zeros(4)

            Q_expected = precision_matrix(model; τ = τ, ρ = ρ)
            @test precision_matrix(gmrf) == Matrix(Q_expected)
        end

        @testset "Type Stability" begin
            model = AR1Model(3)
            Q_f64 = precision_matrix(model; τ = 1.0, ρ = 0.5)
            @test eltype(Q_f64) == Float64
            gmrf = model(τ = 1.0, ρ = 0.5)
            @test gmrf isa GMRF{Float64}
        end

        @testset "Algorithm Storage and Passing" begin
            model = AR1Model(10)
            @test model.alg isa LDLtFactorization
            gmrf = model(τ = 1.0, ρ = 0.5)
            @test gmrf.linsolve_cache.alg isa LDLtFactorization

            custom_model = AR1Model(10, alg = CHOLMODFactorization())
            @test custom_model.alg isa CHOLMODFactorization
            custom_gmrf = custom_model(τ = 1.0, ρ = 0.5)
            @test custom_gmrf.linsolve_cache.alg isa CHOLMODFactorization
        end

        @testset "Custom Constraint" begin
            A_custom = [1.0 0.0 1.0 0.0 0.0]
            e_custom = [2.0]
            model = AR1Model(5, constraint = (A_custom, e_custom))

            A, e = constraints(model)
            @test A ≈ A_custom
            @test e ≈ e_custom

            gmrf = model(τ = 1.0, ρ = 0.5)
            @test gmrf isa ConstrainedGMRF
        end

        @testset "Model name" begin
            @test model_name(AR1Model(5)) == :ar1
        end
    end

    # ============================================================
    # Durbin-Levinson (PACF → AR coefficients)
    # ============================================================
    @testset "Durbin-Levinson" begin
        @testset "P=1" begin
            @test _test_pacf_to_ar([0.8]) ≈ [0.8]
        end

        @testset "P=2" begin
            pacf = [0.8, -0.3]
            phi = _test_pacf_to_ar(pacf)
            # φ₁⁽²⁾ = φ₁⁽¹⁾ - θ₂·φ₁⁽¹⁾ = 0.8 - (-0.3)·0.8 = 1.04, φ₂ = θ₂ = -0.3
            @test phi ≈ [1.04, -0.3]
        end

        @testset "P=3" begin
            phi = _test_pacf_to_ar([0.6, -0.4, 0.2])
            @test length(phi) == 3
        end

        @testset "Zero PACF = white noise" begin
            @test _test_pacf_to_ar([0.0, 0.0]) ≈ [0.0, 0.0]
        end
    end

    # ============================================================
    # ARModel{2}
    # ============================================================
    @testset "ARModel{2}" begin
        @testset "Constructor" begin
            model = ARModel{2}(5)
            @test model.n == 5
            @test model isa ARModel{2}

            @test_throws ArgumentError ARModel{2}(2)
            @test_throws ArgumentError ARModel{2}(1)
            @test_throws ArgumentError ARModel{2}(0)
        end

        @testset "Hyperparameters" begin
            model = ARModel{2}(5)
            params = hyperparameters(model)
            @test haskey(params, :τ)
            @test haskey(params, :pacf1)
            @test haskey(params, :pacf2)
            @test length(params) == 3
        end

        @testset "Parameter Validation" begin
            model = ARModel{2}(5)
            @test_throws ArgumentError precision_matrix(model; τ = 0.0, pacf1 = 0.5, pacf2 = 0.3)
            @test_throws ArgumentError precision_matrix(model; τ = 1.0, pacf1 = 1.0, pacf2 = 0.3)
            @test_throws ArgumentError precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = -1.0)
        end

        @testset "Default solver" begin
            @test ARModel{2}(5).alg isa CHOLMODFactorization
        end

        @testset "Precision Matrix Structure" begin
            @testset "Returns sparse matrix" begin
                model = ARModel{2}(5)
                Q = precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = 0.3)
                @test Q isa SparseMatrixCSC
            end

            @testset "Correctness vs brute-force (n=8)" begin
                n = 8
                model = ARModel{2}(n)
                pacf = [0.7, -0.3]
                τ = 2.0

                Q = precision_matrix(model; τ = τ, pacf1 = pacf[1], pacf2 = pacf[2])

                # Q = τ · σ_e² · Σ⁻¹ where σ_e² = ∏(1 - θ_k²)
                phi = _test_pacf_to_ar(pacf)
                Sigma = _test_ar_autocorrelation_matrix(n, phi)
                sigma_e_sq = prod(1 - θ^2 for θ in pacf)
                Q_expected = τ * sigma_e_sq * inv(Sigma)

                @test Matrix(Q) ≈ Q_expected rtol = 1.0e-10
            end

            @testset "Bandwidth is exactly 2" begin
                model = ARModel{2}(10)
                Q = precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = 0.3)
                Qm = Matrix(Q)
                for i in 1:10, j in 1:10
                    if abs(i - j) > 2
                        @test abs(Qm[i, j]) < 1.0e-14
                    end
                end
            end

            @testset "Symmetry" begin
                model = ARModel{2}(10)
                Q = precision_matrix(model; τ = 1.5, pacf1 = 0.6, pacf2 = -0.4)
                @test issymmetric(Matrix(Q))
            end

            @testset "Positive definiteness" begin
                model = ARModel{2}(10)
                Q = precision_matrix(model; τ = 1.0, pacf1 = 0.8, pacf2 = -0.5)
                evals = eigvals(Symmetric(Matrix(Q)))
                @test all(e -> e > 0, evals)
            end

            @testset "Scaling by τ" begin
                model = ARModel{2}(8)
                Q1 = precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = 0.3)
                Q2 = precision_matrix(model; τ = 3.0, pacf1 = 0.5, pacf2 = 0.3)
                @test Matrix(Q2) ≈ 3.0 * Matrix(Q1)
            end
        end

        @testset "Mean Vector" begin
            @test mean(ARModel{2}(5); τ = 1.0, pacf1 = 0.5, pacf2 = 0.3) == zeros(5)
        end

        @testset "Constraints" begin
            @test constraints(ARModel{2}(5); τ = 1.0, pacf1 = 0.5, pacf2 = 0.3) === nothing
        end

        @testset "GMRF Construction" begin
            model = ARModel{2}(8)
            gmrf = model(τ = 1.0, pacf1 = 0.5, pacf2 = -0.3)
            @test gmrf isa GMRF
            @test length(gmrf) == 8
            @test length(rand(gmrf)) == 8
        end

        @testset "Model name" begin
            @test model_name(ARModel{2}(5)) == :ar2
        end

        @testset "Custom constraint" begin
            model = ARModel{2}(5; constraint = :sumtozero)
            gmrf = model(τ = 1.0, pacf1 = 0.5, pacf2 = -0.3)
            @test gmrf isa ConstrainedGMRF
        end
    end

    # ============================================================
    # ARModel{3}
    # ============================================================
    @testset "ARModel{3}" begin
        @testset "Constructor" begin
            model = ARModel{3}(10)
            @test model.n == 10
            @test model isa ARModel{3}
            @test_throws ArgumentError ARModel{3}(3)
        end

        @testset "Correctness vs brute-force (n=10)" begin
            n = 10
            model = ARModel{3}(n)
            pacf = [0.5, -0.3, 0.2]
            τ = 1.5

            Q = precision_matrix(model; τ = τ, pacf1 = pacf[1], pacf2 = pacf[2], pacf3 = pacf[3])

            phi = _test_pacf_to_ar(pacf)
            Sigma = _test_ar_autocorrelation_matrix(n, phi)
            sigma_e_sq = prod(1 - θ^2 for θ in pacf)
            Q_expected = τ * sigma_e_sq * inv(Sigma)

            @test Matrix(Q) ≈ Q_expected rtol = 1.0e-10
        end

        @testset "Bandwidth is exactly 3" begin
            model = ARModel{3}(10)
            Q = precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = -0.3, pacf3 = 0.2)
            Qm = Matrix(Q)
            for i in 1:10, j in 1:10
                if abs(i - j) > 3
                    @test abs(Qm[i, j]) < 1.0e-14
                end
            end
        end

        @testset "Positive definiteness" begin
            model = ARModel{3}(10)
            Q = precision_matrix(model; τ = 1.0, pacf1 = 0.5, pacf2 = -0.3, pacf3 = 0.2)
            evals = eigvals(Symmetric(Matrix(Q)))
            @test all(e -> e > 0, evals)
        end

        @testset "GMRF Construction" begin
            model = ARModel{3}(10)
            gmrf = model(τ = 1.0, pacf1 = 0.5, pacf2 = -0.3, pacf3 = 0.2)
            @test gmrf isa GMRF
            @test length(gmrf) == 10
        end

        @testset "Model name" begin
            @test model_name(ARModel{3}(10)) == :ar3
        end
    end
end
