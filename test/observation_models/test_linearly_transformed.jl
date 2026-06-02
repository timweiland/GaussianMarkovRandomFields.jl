using LinearAlgebra
using SparseArrays
using ForwardDiff
using Distributions

@testset "LinearlyTransformedObservationModel" begin

    @testset "Basic Construction" begin
        base_model = ExponentialFamily(Poisson)
        A = [1.0 0.5; 1.0 1.0]  # 2 obs, 2 latent components

        ltom = LinearlyTransformedObservationModel(base_model, A)
        @test ltom.base_model === base_model
        @test ltom.design_matrix === A

        # Test hyperparameter delegation
        @test hyperparameters(ltom) == ()  # Poisson has no hyperparameters
    end

    @testset "Materialization" begin
        base_model = ExponentialFamily(Normal)
        A = sparse(1.0 * I, 2, 2)  # Identity
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)

        @test ltlik isa LinearlyTransformedLikelihood
        @test ltlik.design_matrix === A
    end

    @testset "Chain Rule Verification" begin
        base_model = ExponentialFamily(Normal)
        A = sparse([1.0 0.5; 0.0 1.0])
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)
        x_full = [0.5, 1.0]

        # Verify chain rule with ForwardDiff
        grad_fd = ForwardDiff.gradient(x -> loglik(x, ltlik), x_full)
        grad_analytical = loggrad(x_full, ltlik)
        @test grad_analytical ≈ grad_fd

        hess_fd = ForwardDiff.hessian(x -> loglik(x, ltlik), x_full)
        hess_analytical = loghessian(x_full, ltlik)
        @test hess_analytical ≈ hess_fd
    end

    @testset "Identity Matrix Equivalence" begin
        # When A = I, should match base model exactly
        base_model = ExponentialFamily(Poisson)
        A = sparse(1.0 * I, 2, 2)
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = PoissonObservations([1, 3])
        ltlik = ltom(y)
        base_lik = base_model(y)

        x = [0.5, 1.0]
        @test loglik(x, ltlik) ≈ loglik(x, base_lik)
    end

    @testset "Conditional Distribution" begin
        base_model = ExponentialFamily(Poisson)
        A = sparse([1.0 0.5; 0.0 1.0])
        ltom = LinearlyTransformedObservationModel(base_model, A)

        x_full = [0.5, 1.0]
        dist = conditional_distribution(ltom, x_full)

        @test dist isa Distribution
        @test length(dist) == 2

        y = rand(dist)
        @test all(y .>= 0)  # Count data
    end

    @testset "Parameterized design matrix" begin
        base = ExponentialFamily(Normal)
        build_A(; ρ) = sparse([1.0 ρ; 0.0 1.0])

        @testset "Construction, merging, latent_dimension" begin
            spec = ParameterizedMatrix(build_A; hyperparameters = (:ρ,), n_latent = 2)
            ltom = LinearlyTransformedObservationModel(base, spec)
            @test ltom.design_matrix === spec
            @test hyperparameters(ltom) == (:σ, :ρ)          # base-first union
            @test latent_dimension(ltom, [1.0, 2.0]) == 2
        end

        @testset "n_latent defaults to nothing" begin
            spec = ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            ltom = LinearlyTransformedObservationModel(base, spec)
            @test latent_dimension(ltom, [1.0, 2.0]) === nothing
        end

        @testset "Materialization resolves to concrete matrix" begin
            ltom = LinearlyTransformedObservationModel(
                base, ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            )
            ltlik = ltom([1.0, 2.0]; σ = 1.0, ρ = 0.3)
            @test ltlik isa LinearlyTransformedLikelihood
            @test ltlik.design_matrix == build_A(; ρ = 0.3)
            @test ltlik.design_matrix isa SparseMatrixCSC
        end

        @testset "Parameterized == fixed at same ρ" begin
            ρ0 = 0.4
            ltom_fixed = LinearlyTransformedObservationModel(base, build_A(; ρ = ρ0))
            ltom_param = LinearlyTransformedObservationModel(
                base, ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            )
            y = [1.0, 2.0]
            x = [0.5, 1.0]
            lik_fixed = ltom_fixed(y; σ = 1.0)
            lik_param = ltom_param(y; σ = 1.0, ρ = ρ0)
            @test loglik(x, lik_param) ≈ loglik(x, lik_fixed)
            @test loggrad(x, lik_param) ≈ loggrad(x, lik_fixed)
            @test loghessian(x, lik_param) ≈ loghessian(x, lik_fixed)
            # chain rule wrt x still holds for the parameterized likelihood
            @test loggrad(x, lik_param) ≈ ForwardDiff.gradient(z -> loglik(z, lik_param), x)
        end

        @testset "Conjugate Normal path uses resolved matrix" begin
            ρ0 = 0.5
            ltom_fixed = LinearlyTransformedObservationModel(base, build_A(; ρ = ρ0))
            ltom_param = LinearlyTransformedObservationModel(
                base, ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            )
            y = [1.0, 2.0]
            prior = GMRF(zeros(2), sparse(2.0 * I, 2, 2))
            post_fixed = gaussian_approximation(prior, ltom_fixed(y; σ = 1.0))
            post_param = gaussian_approximation(prior, ltom_param(y; σ = 1.0, ρ = ρ0))
            @test mean(post_fixed) ≈ mean(post_param)
        end

        @testset "Conditional distribution with parameterized A" begin
            ltom = LinearlyTransformedObservationModel(
                base, ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            )
            x = [0.5, 1.0]
            d = conditional_distribution(ltom, x; σ = 1.0, ρ = 0.5)
            @test d isa Distribution
            η = build_A(; ρ = 0.5) * x
            @test mean(d) ≈ η
        end

        @testset "Missing hyperparameter errors clearly" begin
            ltom = LinearlyTransformedObservationModel(
                base, ParameterizedMatrix(build_A; hyperparameters = (:ρ,))
            )
            @test_throws ArgumentError ltom([1.0, 2.0]; σ = 1.0)   # ρ missing
        end
    end

    @testset "Affine offset (η = A·x + b)" begin
        base = ExponentialFamily(Normal)
        A = sparse([1.0 0.5; 0.0 1.0])
        y = [1.0, 2.0]
        x = [0.5, 1.0]

        @testset "No offset is backward-compatible" begin
            ltom = LinearlyTransformedObservationModel(base, A)
            ltlik = ltom(y; σ = 1.0)
            @test ltlik.offset === nothing
            @test loglik(x, ltlik) ≈ loglik(A * x, base(y; σ = 1.0))   # η = A·x
            @test hyperparameters(ltom) == (:σ,)
        end

        @testset "Fixed offset" begin
            b = [0.3, -0.2]
            ltom = LinearlyTransformedObservationModel(base, A; offset = b)
            ltlik = ltom(y; σ = 1.0)
            @test ltlik.offset == b
            η = A * x .+ b
            @test loglik(x, ltlik) ≈ loglik(η, base(y; σ = 1.0))
            @test loggrad(x, ltlik) ≈ A' * loggrad(η, base(y; σ = 1.0))
            @test loghessian(x, ltlik) ≈ A' * loghessian(η, base(y; σ = 1.0)) * A
            @test loggrad(x, ltlik) ≈ ForwardDiff.gradient(z -> loglik(z, ltlik), x)
            @test hyperparameters(ltom) == (:σ,)            # fixed offset adds no names
        end

        @testset "ParameterizedOffset materialization + merging" begin
            build_b(; s) = [s, 2s]
            spec = ParameterizedOffset(build_b; hyperparameters = (:s,))
            ltom = LinearlyTransformedObservationModel(base, A; offset = spec)
            @test hyperparameters(ltom) == (:σ, :s)         # base, then offset
            ltlik = ltom(y; σ = 1.0, s = 0.5)
            @test ltlik.offset == build_b(; s = 0.5)
            @test loglik(x, ltlik) ≈ loglik(A * x .+ build_b(; s = 0.5), base(y; σ = 1.0))
        end

        @testset "Parameterized offset == fixed at same θ" begin
            build_b(; s) = [s, 2s]
            s0 = 0.4
            lf = LinearlyTransformedObservationModel(base, A; offset = build_b(; s = s0))(y; σ = 1.0)
            lp = LinearlyTransformedObservationModel(
                base, A; offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
            )(y; σ = 1.0, s = s0)
            @test loglik(x, lp) ≈ loglik(x, lf)
            @test loggrad(x, lp) ≈ loggrad(x, lf)
            @test loghessian(x, lp) ≈ loghessian(x, lf)
        end

        @testset "Conjugate Normal path shifts mean by offset" begin
            b = [0.5, -0.3]
            prior = GMRF(zeros(2), sparse(2.0 * I, 2, 2))
            post = gaussian_approximation(prior, LinearlyTransformedObservationModel(base, A; offset = b)(y; σ = 1.0))
            # Equivalent: no offset, conditioning on shifted data y − b.
            post0 = gaussian_approximation(prior, LinearlyTransformedObservationModel(base, A)(y .- b; σ = 1.0))
            @test mean(post) ≈ mean(post0)
        end

        @testset "Errors" begin
            build_b(; s) = [s, 2s]
            ltom = LinearlyTransformedObservationModel(
                base, A; offset = ParameterizedOffset(build_b; hyperparameters = (:s,))
            )
            @test_throws ArgumentError ltom(y; σ = 1.0)               # s missing
            # offset length must match the number of observations
            @test_throws DimensionMismatch LinearlyTransformedObservationModel(base, A; offset = [1.0, 2.0, 3.0])(y; σ = 1.0)
        end
    end

end
