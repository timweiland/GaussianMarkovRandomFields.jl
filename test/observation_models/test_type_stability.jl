using Distributions
using ForwardDiff
using DifferentiationInterface: AutoForwardFromPrimitive
using DifferentiationInterface

@testset "ObsModel Type Stability" begin

    @testset "ExponentialFamily Type Stability" begin
        # Test that ObservationLikelihood models are type stable using new API

        # Poisson model
        poisson_model = ExponentialFamily(Poisson)
        x = [1.0, 2.0]
        y = [1, 3]
        poisson_lik = poisson_model(y)

        @inferred loglik(x, poisson_lik)
        @inferred loggrad(x, poisson_lik)
        @inferred loghessian(x, poisson_lik)

        # Bernoulli model
        bernoulli_model = ExponentialFamily(Bernoulli)
        y_bool = [0, 1]
        bernoulli_lik = bernoulli_model(y_bool)

        @inferred loglik(x, bernoulli_lik)
        @inferred loggrad(x, bernoulli_lik)
        @inferred loghessian(x, bernoulli_lik)

        # Normal model
        normal_model = ExponentialFamily(Normal)
        y_float = [1.1, 2.2]
        normal_lik = normal_model(y_float; σ = 1.0)

        @inferred loglik(x, normal_lik)
        @inferred loggrad(x, normal_lik)
        @inferred loghessian(x, normal_lik)
    end

    @testset "Link Function Type Stability" begin
        # Test link functions
        @inferred apply_link(IdentityLink(), 1.0)
        @inferred apply_invlink(IdentityLink(), 1.0)
        @inferred GaussianMarkovRandomFields.derivative_invlink(IdentityLink(), 1.0)
        @inferred GaussianMarkovRandomFields.second_derivative_invlink(IdentityLink(), 1.0)

        @inferred apply_link(LogLink(), 1.0)
        @inferred apply_invlink(LogLink(), 1.0)
        @inferred GaussianMarkovRandomFields.derivative_invlink(LogLink(), 1.0)
        @inferred GaussianMarkovRandomFields.second_derivative_invlink(LogLink(), 1.0)

        @inferred apply_link(LogitLink(), 0.5)
        @inferred apply_invlink(LogitLink(), 0.0)
        @inferred GaussianMarkovRandomFields.derivative_invlink(LogitLink(), 0.0)
        @inferred GaussianMarkovRandomFields.second_derivative_invlink(LogitLink(), 0.0)

        # Test broadcasting
        x = [1.0, 2.0, 3.0]
        @inferred (() -> apply_link.(Ref(IdentityLink()), x))()
        @inferred (() -> apply_invlink.(Ref(LogLink()), x))()
        @inferred (() -> GaussianMarkovRandomFields.derivative_invlink.(Ref(LogitLink()), x))()
    end

    @testset "Custom ObservationLikelihood Type Stability" begin
        # Simple custom likelihood for testing
        struct SimpleCustomLikelihood <: ObservationLikelihood
            y::Vector{Float64}
        end

        function GaussianMarkovRandomFields.loglik(x, lik::SimpleCustomLikelihood)
            return -0.5 * sum((x .- lik.y) .^ 2)
        end
        GaussianMarkovRandomFields.autodiff_gradient_backend(::SimpleCustomLikelihood) = AutoForwardDiff()
        GaussianMarkovRandomFields.autodiff_hessian_backend(::SimpleCustomLikelihood) = AutoForwardDiff()

        y = [1.1, 2.1]
        lik = SimpleCustomLikelihood(y)
        x = [1.0, 2.0]

        @inferred loglik(x, lik)
        # Note: AD fallbacks may not be type stable due to ForwardDiff internals,
        # but we can still test that they return correct types
        grad_result = loggrad(x, lik)
        hess_result = loghessian(x, lik)

        @test grad_result isa Vector{Float64}
        @test hess_result isa AbstractMatrix{Float64}
    end

    @testset "Parametric Type Consistency" begin
        # Test that parametric types work as expected
        poisson_log = ExponentialFamily(Poisson, LogLink())
        poisson_identity = ExponentialFamily(Poisson, IdentityLink())

        @test typeof(poisson_log) != typeof(poisson_identity)
        @test poisson_log.family == poisson_identity.family
        @test typeof(poisson_log.link) != typeof(poisson_identity.link)

        # Both should be type stable with new API
        x = [1.0, 2.0]
        y = [1, 2]

        poisson_log_lik = poisson_log(y)
        poisson_identity_lik = poisson_identity(y)

        @inferred loglik(x, poisson_log_lik)
        @inferred loglik(x, poisson_identity_lik)
    end
end
