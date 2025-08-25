using Distributions
using ForwardDiff
using LinearAlgebra
using StatsFuns
using Random

# Define test RNG
const test_rng = MersenneTwister(123)

# Helper function to test gradient and hessian against ForwardDiff using new API
function test_against_autodiff(obs_lik, η)
    # Test gradient
    grad = loggrad(η, obs_lik)
    grad_fd = ForwardDiff.gradient(x -> loglik(x, obs_lik), η)
    @test grad ≈ grad_fd atol = 1.0e-4

    # Test hessian
    hess = loghessian(η, obs_lik)
    hess_fd = ForwardDiff.hessian(x -> loglik(x, obs_lik), η)
    return @test Matrix(hess) ≈ hess_fd atol = 1.0e-4
end

@testset "ExponentialFamily Models" begin

    @testset "Poisson Family" begin
        # Test with different link functions
        links = [LogLink(), IdentityLink()]

        for link in links
            model = ExponentialFamily(Poisson, link)

            # Generate appropriate test data for this link
            if link isa LogLink
                η = randn(5)  # Any real values work for log link
            else  # IdentityLink
                η = abs.(randn(5)) .+ 0.1  # Must be positive for Poisson rates
            end

            y = rand(0:10, 5)  # Random count data

            # Create materialized likelihood using new API
            obs_lik = model(y)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Bernoulli Family" begin
        # Test with different link functions
        links = [LogitLink(), LogLink()]

        for link in links
            model = ExponentialFamily(Bernoulli, link)

            # Generate appropriate test data for this link
            if link isa LogitLink
                η = randn(5)  # Any real values work for logit link
            else  # LogLink
                η = log.(rand(5) .* 0.8 .+ 0.1)  # log of probabilities in (0,1)
            end

            y = rand([0, 1], 5)  # Random binary data

            # Create materialized likelihood using new API
            obs_lik = model(y)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Binomial Family" begin
        # Test with different link functions
        links = [LogitLink(), IdentityLink()]
        n = 10  # Fixed number of trials

        for link in links
            model = ExponentialFamily(Binomial, link)

            # Generate appropriate test data for this link
            if link isa LogitLink
                η = randn(5)  # Any real values work for logit link
            else  # IdentityLink
                η = rand(5) .* 0.8 .+ 0.1  # Probabilities in (0,1)
            end

            y = BinomialObservations(rand(0:n, 5), repeat([n], 5))

            # Create materialized likelihood using new API
            obs_lik = model(y; n = n)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Normal Family" begin
        # Test with different link functions
        links = [IdentityLink(), LogLink()]
        σ = 0.5  # Fixed standard deviation

        for link in links
            model = ExponentialFamily(Normal, link)

            # Generate appropriate test data for this link
            if link isa IdentityLink
                η = randn(5)  # Any real values work for identity link
            else  # LogLink
                η = log.(abs.(randn(5)) .+ 0.1)  # log of positive values
            end

            y = randn(5)  # Random normal data

            # Create materialized likelihood using new API
            obs_lik = model(y; σ = σ)

            test_against_autodiff(obs_lik, η)
        end
    end

    @testset "Indexed ExponentialFamily Support" begin
        @testset "Constructor with indices parameter" begin
            # Test ExponentialFamily constructor with indices parameter
            gaussian_model = ExponentialFamily(Normal, indices = 1:3)
            @test gaussian_model isa ExponentialFamily
            @test gaussian_model.indices == 1:3

            # Test with different index types
            poisson_model = ExponentialFamily(Poisson, indices = [1, 3, 5])
            @test poisson_model.indices == [1, 3, 5]

            # Test default behavior (no indices - should target all)
            default_model = ExponentialFamily(Normal)
            @test !hasfield(typeof(default_model), :indices) || default_model.indices === nothing
        end

        @testset "Hyperparameters with indexed models" begin
            # Indexed models should have same hyperparameters as non-indexed
            indexed_normal = ExponentialFamily(Normal, indices = 1:5)
            regular_normal = ExponentialFamily(Normal)

            @test hyperparameters(indexed_normal) == hyperparameters(regular_normal)
            @test hyperparameters(indexed_normal) == (:σ,)

            indexed_poisson = ExponentialFamily(Poisson, indices = 2:4)
            regular_poisson = ExponentialFamily(Poisson)

            @test hyperparameters(indexed_poisson) == hyperparameters(regular_poisson)
            @test hyperparameters(indexed_poisson) == ()
        end

        @testset "Factory callable with indices" begin
            # Test that indexed models can be materialized
            indexed_model = ExponentialFamily(Normal, indices = 2:4)
            y = [1.0, 2.0, 3.0]  # 3 observations for indices 2:4

            # Should create indexed likelihood
            indexed_lik = indexed_model(y; σ = 1.5)
            @test indexed_lik isa NormalLikelihood  # or whatever the materialized type is

            # The materialized likelihood should know about indices
            @test hasfield(typeof(indexed_lik), :indices)
            @test indexed_lik.indices == 2:4
        end

        @testset "Indexed evaluation" begin
            # Test that indexed likelihoods only use specified indices
            indexed_model = ExponentialFamily(Normal, indices = 2:3)
            y = [1.0, 2.0]  # 2 observations for indices 2:3

            indexed_lik = indexed_model(y; σ = 1.0)

            # Should evaluate correctly with 4D latent field
            x = [10.0, 20.0, 30.0, 40.0]  # Only indices 2:3 should matter
            ll = loglik(x, indexed_lik)

            # Should equal evaluation on just the relevant indices
            regular_model = ExponentialFamily(Normal)
            regular_lik = regular_model(y; σ = 1.0)
            ll_expected = loglik(x[2:3], regular_lik)

            @test ll ≈ ll_expected
        end
    end

    @testset "Indexed Exponential Family Evaluation" begin
        # Test all families with indices - comprehensive evaluation testing
        families_and_links = [
            (Normal, [IdentityLink(), LogLink()]),
            (Poisson, [LogLink(), IdentityLink()]),
            (Bernoulli, [LogitLink(), LogLink()]),
            (Binomial, [LogitLink(), IdentityLink()]),
        ]

        for (family, links) in families_and_links
            @testset "$(family) with indices" begin
                for link in links
                    # Create indexed model targeting indices 2:4 out of 6D latent field
                    indexed_model = ExponentialFamily(family, link, indices = 2:4)

                    # Generate appropriate test data for this link and family
                    if link isa IdentityLink && family == Normal
                        η_full = randn(6)  # Any real values work for identity link
                    elseif link isa LogLink && (family == Normal || family == Poisson)
                        η_full = log.(abs.(randn(6)) .+ 0.1)  # log of positive values
                    elseif link isa LogLink && (family == Bernoulli || family == Binomial)
                        η_full = log.(rand(6) .* 0.8 .+ 0.1)  # log of probabilities in (0,1)
                    elseif link isa LogitLink
                        η_full = randn(6)  # Any real values work for logit link
                    elseif link isa IdentityLink && family == Poisson
                        η_full = abs.(randn(6)) .+ 0.1  # Positive values for Poisson rate
                    elseif link isa IdentityLink && (family == Bernoulli || family == Binomial)
                        η_full = rand(6) .* 0.8 .+ 0.1  # Probabilities in (0,1) for identity link
                    else
                        error("Unexpected combination: $(family) with $(link)")
                    end

                    # Generate observation data (3 observations for indices 2:4)
                    if family == Normal
                        y = randn(3)
                        σ = 0.5 + rand()
                        indexed_lik = indexed_model(y; σ = σ)
                    elseif family == Poisson
                        y = rand(0:10, 3)
                        indexed_lik = indexed_model(y)
                    elseif family == Bernoulli
                        y = rand(0:1, 3)
                        indexed_lik = indexed_model(y)
                    else  # Binomial
                        n = 10
                        y = BinomialObservations(rand(0:n, 3), repeat([n], 3))
                        indexed_lik = indexed_model(y; n = n)
                    end

                    # Test loglik evaluation
                    ll_indexed = loglik(η_full, indexed_lik)
                    @test ll_indexed isa Float64
                    @test isfinite(ll_indexed)

                    # Compare against non-indexed version using only relevant indices
                    regular_model = ExponentialFamily(family, link)
                    if family == Normal
                        regular_lik = regular_model(y; σ = σ)
                    elseif family == Poisson
                        regular_lik = regular_model(y)
                    elseif family == Bernoulli
                        regular_lik = regular_model(y)
                    else  # Binomial
                        regular_lik = regular_model(y; n = n)
                    end

                    ll_regular = loglik(η_full[2:4], regular_lik)
                    @test ll_indexed ≈ ll_regular

                    # Test gradient evaluation
                    grad_indexed = loggrad(η_full, indexed_lik)
                    @test length(grad_indexed) == 6  # Same length as input
                    @test all(isfinite, grad_indexed)

                    # Check that gradient is zero outside indexed region
                    @test grad_indexed[1] == 0.0
                    @test grad_indexed[5] == 0.0
                    @test grad_indexed[6] == 0.0

                    # Check that gradient matches non-indexed version on indexed region
                    grad_regular = loggrad(η_full[2:4], regular_lik)
                    @test grad_indexed[2:4] ≈ grad_regular

                    # Test hessian evaluation
                    hess_indexed = loghessian(η_full, indexed_lik)
                    @test size(hess_indexed) == (6, 6)
                    @test all(isfinite, hess_indexed)

                    # Check that Hessian is zero outside indexed region
                    @test all(hess_indexed[1, :] .== 0.0)
                    @test all(hess_indexed[:, 1] .== 0.0)
                    @test all(hess_indexed[5, :] .== 0.0)
                    @test all(hess_indexed[:, 5] .== 0.0)
                    @test all(hess_indexed[6, :] .== 0.0)
                    @test all(hess_indexed[:, 6] .== 0.0)

                    # Check that Hessian matches non-indexed version on indexed region
                    hess_regular = loghessian(η_full[2:4], regular_lik)
                    @test Matrix(hess_indexed[2:4, 2:4]) ≈ Matrix(hess_regular)

                    # Test against ForwardDiff for correctness
                    grad_fd = ForwardDiff.gradient(x -> loglik(x, indexed_lik), η_full)
                    @test grad_indexed ≈ grad_fd atol = 1.0e-4

                    hess_fd = ForwardDiff.hessian(x -> loglik(x, indexed_lik), η_full)
                    @test Matrix(hess_indexed) ≈ hess_fd atol = 1.0e-4
                end
            end
        end
    end

    @testset "Data Distribution Interface" begin
        # Test data_distribution function for all families
        @testset "Normal data distribution" begin
            model = ExponentialFamily(Normal)
            x = [0.5, -0.2, 1.1]
            θ_named = (σ = 0.8,)

            dist = data_distribution(model, x, θ_named)
            @test dist isa Distribution
            @test length(dist) == 3

            # Sample from distribution
            y = rand(test_rng, dist)
            @test length(y) == 3
        end

        @testset "Poisson data distribution" begin
            model = ExponentialFamily(Poisson)
            x = [0.5, 1.2, -0.1]  # Log scale
            θ_named = NamedTuple()

            dist = data_distribution(model, x, θ_named)
            @test dist isa Distribution
            @test length(dist) == 3

            # Sample from distribution
            y = rand(test_rng, dist)
            @test length(y) == 3
            @test all(y .>= 0)  # Count data
        end

        @testset "Bernoulli data distribution" begin
            model = ExponentialFamily(Bernoulli)
            x = [0.0, 1.5, -0.8]  # Logit scale
            θ_named = NamedTuple()

            dist = data_distribution(model, x, θ_named)
            @test dist isa Distribution
            @test length(dist) == 3

            # Sample from distribution
            y = rand(test_rng, dist)
            @test length(y) == 3
            @test all(y .∈ Ref([0, 1]))  # Binary data
        end

        @testset "Binomial data distribution" begin
            model = ExponentialFamily(Binomial)
            x = [0.2, -0.5, 1.0]  # Logit scale
            θ_named = (n = 10,)

            dist = data_distribution(model, x, θ_named)
            @test dist isa Distribution
            @test length(dist) == 3

            # Sample from distribution
            y = rand(test_rng, dist)
            @test length(y) == 3
            @test all(0 .<= y .<= 10)  # Count data bounded by n
        end
    end

    @testset "Hyperparameters Interface" begin
        # Test hyperparameters function for families not covered elsewhere
        @testset "Bernoulli hyperparameters" begin
            model = ExponentialFamily(Bernoulli)
            @test hyperparameters(model) == ()
        end

        @testset "Binomial hyperparameters" begin
            model = ExponentialFamily(Binomial)
            @test hyperparameters(model) == ()
        end
    end

    @testset "Latent Dimension Interface" begin
        # Test latent_dimension function
        model = ExponentialFamily(Normal)
        y = [1.0, 2.0, 3.0, 4.0]
        @test latent_dimension(model, y) == 4

        # Test with different observation lengths
        y_short = [1.0, 2.0]
        @test latent_dimension(model, y_short) == 2
    end

    @testset "Random Sampling Interface" begin
        # Test Random.rand method for ExponentialFamily
        @testset "Normal sampling" begin
            model = ExponentialFamily(Normal)
            x = [0.5, -0.2, 1.1]
            θ_named = (σ = 0.8,)

            y = rand(test_rng, model; x = x, θ_named = θ_named)
            @test length(y) == 3
            @test all(isfinite, y)
        end

        @testset "Poisson sampling" begin
            model = ExponentialFamily(Poisson)
            x = [0.5, 1.2, -0.1]  # Log scale
            θ_named = NamedTuple()

            y = rand(test_rng, model; x = x, θ_named = θ_named)
            @test length(y) == 3
            @test all(y .>= 0)  # Count data
        end

        @testset "Bernoulli sampling" begin
            model = ExponentialFamily(Bernoulli)
            x = [0.0, 1.5, -0.8]  # Logit scale
            θ_named = NamedTuple()

            y = rand(test_rng, model; x = x, θ_named = θ_named)
            @test length(y) == 3
            @test all(y .∈ Ref([0, 1]))  # Binary data
        end
    end
end
