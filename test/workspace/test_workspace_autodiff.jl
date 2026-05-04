using GaussianMarkovRandomFields
using Distributions: logpdf, Normal, Poisson
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, Zygote, ForwardDiff

function ar_precision_sparse(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Pipeline: θ → WorkspaceGMRF → logpdf
function test_workspace_logpdf_pipeline(θ::Vector, z::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    gmrf = WorkspaceGMRF(μ, Q)
    return logpdf(gmrf, z)
end

# Pipeline: θ → WorkspaceGMRF → gaussian_approximation → logpdf
function test_workspace_ga_pipeline(θ::Vector, y::Vector, x::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    prior = WorkspaceGMRF(μ, Q)
    obs_model = ExponentialFamily(Poisson)
    obs_lik = obs_model(PoissonObservations(y))
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, x)
end

# Reference pipeline with standard GMRF (for cross-check)
function test_gmrf_logpdf_pipeline(θ::Vector, z::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    gmrf = GMRF(μ, Q)
    return logpdf(gmrf, z)
end

function test_gmrf_ga_pipeline(θ::Vector, y::Vector, x::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    prior = GMRF(μ, Q)
    obs_model = ExponentialFamily(Poisson)
    obs_lik = obs_model(PoissonObservations(y))
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, x)
end

backends = Any[("Zygote", AutoZygote()), ("ForwardDiff", AutoForwardDiff())]

@testset "$backend_name WorkspaceGMRF autodiff" for (backend_name, backend) in backends
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    @testset "logpdf gradient" begin
        k = 10
        θ = [0.5, 0.1]
        z = randn(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            backend, θ
        )
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            fd_backend, θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "logpdf gradient matches GMRF" begin
        k = 10
        θ = [0.5, 0.1]
        z = randn(k)

        grad_ws = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            backend, θ
        )
        grad_gmrf = DifferentiationInterface.gradient(
            θ -> test_gmrf_logpdf_pipeline(θ, z, k),
            backend, θ
        )

        @test grad_ws ≈ grad_gmrf rtol = 1.0e-8
    end

    @testset "GA + logpdf gradient" begin
        k = 8
        θ = [0.5, 0.0]
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            backend, θ
        )
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            fd_backend, θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 2.0e-2
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "GA + logpdf gradient matches GMRF" begin
        k = 8
        θ = [0.5, 0.0]
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)

        grad_ws = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            backend, θ
        )
        grad_gmrf = DifferentiationInterface.gradient(
            θ -> test_gmrf_ga_pipeline(θ, y, x, k),
            backend, θ
        )

        @test grad_ws ≈ grad_gmrf rtol = 1.0e-4
    end

end

@testset "ForwardDiff WorkspaceGMRF reuse path" begin
    # Exercises the `make_workspace(m) -> model(ws; θ...) -> logpdf` pattern
    # that downstream hyperparameter-inference consumers use. Workspace is
    # built once outside the differentiated function so its factorization
    # survives across calls.
    #
    # Note: ForwardDiff-only for now. Zygote on the LatentModel-callable
    # reuse path requires an rrule on the callable itself (separate gap;
    # tracked as future work).
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    @testset "Unconstrained logpdf gradient" begin
        k = 10
        θ = [0.5, 0.1]
        z = randn(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        pipeline = θ -> logpdf(model(ws; τ = exp(θ[1]), ρ = tanh(θ[2])), z)

        grad_test = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "Unconstrained gaussian_approximation gradient" begin
        # Same reuse pattern but with a non-Gaussian observation likelihood
        # going through gaussian_approximation.
        k = 8
        θ = [0.5, 0.0]
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)

        function pipeline(θ)
            prior = model(ws; τ = exp(θ[1]), ρ = tanh(θ[2]))
            obs_model = ExponentialFamily(Poisson)
            obs_lik = obs_model(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        grad_test = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 2.0e-2
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Float64 WorkspaceGMRF + Dual obs_lik gaussian_approximation" begin
        # Regression for SHOULD-FIX #7: when the prior is a Float64
        # WorkspaceGMRF and only the observation likelihood carries Duals
        # (e.g. differentiating through σ in a Normal likelihood),
        # `gaussian_approximation` must take the obs-dual workspace path,
        # which runs the primal Newton solve once and propagates Dual
        # tangents via the IFT.
        k = 8
        y = randn(k) .* 0.3 .+ 0.2
        x = randn(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)  # WorkspaceGMRF{Float64}

        function pipeline(θ)
            obs_lik = ExponentialFamily(Normal)(y; σ = exp(θ[1]))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [log(0.5)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Constrained Float64 WorkspaceGMRF + Dual obs_lik" begin
        # Same regression for the constrained branch of the obs-dual
        # workspace path: RW1 prior (sum-to-zero) with Float64 hyperparameters
        # and a Normal obs_lik whose σ depends on θ.
        k = 8
        y = randn(k) .* 0.3
        x_eval = randn(k); x_eval .-= sum(x_eval) / k
        model = RW1Model(k)
        ws = make_workspace(model; τ = 1.0)
        prior = model(ws; τ = 1.0)  # constrained WorkspaceGMRF{Float64}

        function pipeline(θ)
            obs_lik = ExponentialFamily(Normal)(y; σ = exp(θ[1]))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(0.5)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Float64 WorkspaceGMRF + AutoDiffLikelihood Dual hyperparams" begin
        # IFT path for AutoDiffLikelihood with a Dual scalar hyperparameter:
        # the closure remains primal, hyperparams are stored on the likelihood,
        # and the FD-extension `gaussian_approximation` dispatch detects the
        # Dual and runs primal-Newton + per-partial θ-tangent FD.
        Random.seed!(7)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)  # WorkspaceGMRF{Float64}

        function poisson_loglik(x; y, φ)
            return sum(y .* (φ .* x) .- exp.(φ .* x))
        end
        obs_model = AutoDiffObservationModel(
            poisson_loglik;
            n_latent = k,
            hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            obs_lik = obs_model(y; φ = exp(θ[1]))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [0.0]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Constrained Float64 WorkspaceGMRF + AutoDiffLikelihood Dual hyperparams" begin
        # Same IFT path through a constrained prior (RW1 sum-to-zero).
        Random.seed!(11)
        k = 8
        y = randn(k) .* 0.3
        x_eval = randn(k); x_eval .-= sum(x_eval) / k
        model = RW1Model(k)
        ws = make_workspace(model; τ = 1.0)
        prior = model(ws; τ = 1.0)  # constrained WorkspaceGMRF{Float64}

        function gauss_loglik(x; y, φ)
            return -0.5 * sum((y .- x) .^ 2) * exp(2 * φ) + length(y) * φ
        end
        obs_model = AutoDiffObservationModel(
            gauss_loglik;
            n_latent = k,
            hyperparams = (:φ,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            obs_lik = obs_model(y; φ = θ[1])
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [log(2.0)]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Multi-hyperparam + length(θ)>1 outer gradient" begin
        # IFT path with two named scalar hyperparameters, both carrying
        # Duals from an outer ForwardDiff.gradient over a length-2 θ.
        # Exercises N=2 partials in the IFT-loop and the multi-hp perturb
        # path that single-hyperparam tests don't.
        Random.seed!(17)
        k = 8
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        function poisson_loglik2(x; y, φ, β)
            return sum(y .* (φ .* x .+ β) .- exp.(φ .* x .+ β))
        end
        obs_model = AutoDiffObservationModel(
            poisson_loglik2;
            n_latent = k,
            hyperparams = (:φ, :β),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            obs_lik = obs_model(y; φ = exp(θ[1]), β = θ[2])
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [0.0, 0.1]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Matrix-valued Dual hyperparam (shape preservation)" begin
        # Regression for a bug in `_perturb_one(::AbstractArray{<:Dual})`
        # where the partials were flattened to a Vector via comprehension,
        # breaking broadcast against same-shape values for non-vector
        # hyperparams. Use a small Matrix hyperparam.
        Random.seed!(13)
        k = 6
        y = randn(k) .* 0.3
        x = zeros(k)
        model = AR1Model(k)
        ws = make_workspace(model; τ = 1.0, ρ = 0.3)
        prior = model(ws; τ = 1.0, ρ = 0.3)

        # Loglik that uses a 2x2 hyperparam matrix W in a way that contracts
        # to a scalar. The pipeline differentiates through trace(W).
        function mat_loglik(x; y, W)
            s = sum(W) / size(W, 1)
            return -0.5 * sum((y .- x) .^ 2) * s
        end
        obs_model = AutoDiffObservationModel(
            mat_loglik;
            n_latent = k,
            hyperparams = (:W,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )

        function pipeline(θ)
            W = [θ[1] 0.0; 0.0 θ[1]]
            obs_lik = obs_model(y; W = W)
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x)
        end

        θ = [1.0]
        grad_fwd = DifferentiationInterface.gradient(pipeline, AutoForwardDiff(), θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)

        abs_error = abs.(grad_fwd - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end
end
