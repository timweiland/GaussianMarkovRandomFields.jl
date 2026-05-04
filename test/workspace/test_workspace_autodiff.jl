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
        # Dual and runs primal-Newton + exact-AD θ-tangents (no FD).
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

        # Tight thresholds — AD-on-θ should match FD up to FD's own discretization
        # error (~1e-7 with AutoFiniteDiff defaults).
        abs_error = abs.(grad_fwd - grad_fd)
        @test maximum(abs_error) < 1.0e-4
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
        @test maximum(abs_error) < 1.0e-4
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
        @test maximum(abs_error) < 1.0e-4
    end

    @testset "Analytic AD-on-θ cross-check (Gaussian loglik)" begin
        # Validates the AD-on-θ partial extraction at machine precision
        # against a closed-form analytic derivative. Uses a Gaussian loglik
        # whose Hessian and Hessian-θ-derivative are known exactly.
        #
        #   loglik(x; y, α) = -0.5 * α * Σ(y_i - x_i)²
        #     g_i   = ∂loglik/∂x_i  = α * (y_i - x_i)
        #     ∂g_i/∂α at fixed x    = y_i - x_i
        #     H_ii  = ∂²loglik/∂x_i² = -α
        #     ∂H_ii/∂α at fixed x   = -1
        #     ∂H_ii/∂x_j            = 0 → total dH/dα equals ∂H/∂α regardless of dx/dα
        #
        # This is a direct AD-machinery test — bypasses gaussian_approximation
        # so the only error sources are AD itself (which is exact) and
        # floating-point. The 1e-12 threshold pins this.
        function gauss_α(x; y, α)
            return -0.5 * α * sum((y .- x) .^ 2)
        end
        n_latent = 4
        obs_model = AutoDiffObservationModel(
            gauss_α;
            n_latent = n_latent,
            hyperparams = (:α,),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        y_data = [1.0, 2.0, 3.0, 4.0]
        x_test = [0.5, 1.5, 2.5, 3.5]

        # Driver function that returns sum of grad partials so outer
        # ForwardDiff.derivative reduces to an analytic-comparable scalar.
        # Inside, it exercises exactly the AD-on-θ lift we use in
        # `_autodifflik_ift_workspace` step 2.
        function grad_dα_sum(α)
            obs_lik = obs_model(y_data; α = α)
            φ = obs_lik.hyperparams.α
            if φ isa ForwardDiff.Dual
                DT = typeof(φ)
                N = ForwardDiff.npartials(DT)
                zp = ForwardDiff.Partials{N, Float64}(ntuple(_ -> 0.0, Val(N)))
                x_lifted = [DT(x_test[i], zp) for i in 1:n_latent]
                g = loggrad(x_lifted, obs_lik)
                # Reduce to scalar via sum so outer FD.derivative returns Float64.
                return sum(g)
            else
                return sum(loggrad(x_test, obs_lik))
            end
        end

        α_val = 2.0
        # Outer FD.derivative passes Dual α; AD-on-θ path returns sum(g) as
        # Dual; outer derivative reads partial.
        ad_partial = ForwardDiff.derivative(grad_dα_sum, α_val)
        # Analytic Σᵢ ∂gᵢ/∂α at fixed x = Σᵢ (yᵢ - xᵢ)
        analytic_partial = sum(y_data .- x_test)
        @test abs(ad_partial - analytic_partial) < 1.0e-12

        # Hessian total derivative — same loglik. dH/dα at fixed x = -n
        # (since H = -α I and dH_ii/dα = -1 with no x-dependence).
        function hess_dα_sum(α)
            obs_lik = obs_model(y_data; α = α)
            φ = obs_lik.hyperparams.α
            if φ isa ForwardDiff.Dual
                DT = typeof(φ)
                N = ForwardDiff.npartials(DT)
                # Use arbitrary dx/dα — H is x-independent so total = ∂H/∂α.
                dx_dα = [0.3, -0.7, 0.1, 0.05]
                x_lifted = [
                    DT(x_test[i], ForwardDiff.Partials{N, Float64}((dx_dα[i],)))
                        for i in 1:n_latent
                ]
                H = loghessian(x_lifted, obs_lik)
                return sum(diag(H))
            else
                H = loghessian(x_test, obs_lik)
                return sum(diag(H))
            end
        end
        ad_hess_partial = ForwardDiff.derivative(hess_dα_sum, α_val)
        # Analytic: dH_ii/dα = -1 for each of n_latent diagonal entries
        analytic_hess_partial = -float(n_latent)
        @test abs(ad_hess_partial - analytic_hess_partial) < 1.0e-12
    end

    @testset "Mismatched outer-Dual tags errors loudly" begin
        # Defensive guard: if hyperparams carry Duals from different outer
        # AD passes (different Tags), `_outer_tag_and_npartials` should
        # error rather than silently misread partials.
        function loglik2(x; y, α, β)
            return sum(α .* x .+ β .* x .^ 2)
        end
        obs_model = AutoDiffObservationModel(
            loglik2;
            n_latent = 3,
            hyperparams = (:α, :β),
            grad_backend = AutoForwardDiff(),
            hessian_backend = AutoForwardDiff(),
        )
        y = [1.0, 2.0, 3.0]
        # Manually build two Duals with DISTINCT tags to simulate the
        # "two independent outer passes" misuse case.
        TagA = ForwardDiff.Tag{Symbol("A"), Float64}
        TagB = ForwardDiff.Tag{Symbol("B"), Float64}
        α_d = ForwardDiff.Dual{TagA, Float64, 1}(0.5, ForwardDiff.Partials{1, Float64}((1.0,)))
        β_d = ForwardDiff.Dual{TagB, Float64, 1}(0.3, ForwardDiff.Partials{1, Float64}((1.0,)))
        obs_lik = obs_model(y; α = α_d, β = β_d)
        prior = WorkspaceGMRF(zeros(3), spdiagm(0 => ones(3)))
        @test_throws ErrorException gaussian_approximation(prior, obs_lik)
    end

    @testset "Matrix-valued Dual hyperparam (shape preservation)" begin
        # Regression for shape preservation when extracting partials from a
        # Matrix-valued hyperparameter. Originally this exercised an
        # FD-on-θ helper that flattened partials via list comprehension; the
        # AD-on-θ rewrite uses ForwardDiff.partials directly, but the test
        # still pins the contract that Matrix hyperparams round-trip cleanly.
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
        @test maximum(abs_error) < 1.0e-4
    end
end
