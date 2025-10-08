#!/usr/bin/env julia
"""
Benchmark: Compare AD backends for high-dimensional hyperparameter optimization

Compares FiniteDiff, Zygote, and Enzyme for computing gradients w.r.t. ~100 hyperparameters
through a typical GMRF workflow: hyperparameters → GMRF → gaussian_approximation → logpdf

Usage:
    cd benchmarks
    julia --project=. autodiff_comparison.jl
"""

using GaussianMarkovRandomFields
using DifferentiationInterface
using BenchmarkTools
using Distributions
using SparseArrays
using LinearAlgebra
using LinearSolve
using Printf
using Random

using Zygote, Enzyme, FiniteDiff

println("="^80)
println("AUTODIFF BACKEND COMPARISON: HIGH-DIMENSIONAL HYPERPARAMETER SPACE")
println("="^80)

# Problem setup: RW1 temporal model with location-specific means
n = 100  # Number of time points/locations
n_params = n + 1  # n mean parameters + 1 precision parameter

println("\nProblem setup:")
println("  RW1 temporal model with $n time points")
println("  Hyperparameters: $(n_params) total")
println("    - $n mean parameters (one per time point)")
println("    - 1 precision parameter (τ)")

# Workflow: θ → GMRF → gaussian_approximation → logpdf
function benchmark_workflow(θ::Vector{Float64}, y::Vector{Int}, x_eval::Vector{Float64})
    # Extract hyperparameters
    μ = θ[1:n]      # Mean field (100 params)
    log_τ = θ[n + 1]  # Log precision
    τ = exp(log_τ)

    # Build precision matrix using RW1 latent model
    rw1_model = RW1Model(n)
    Q = precision_matrix(rw1_model; τ = τ)

    # Prior GMRF with custom mean
    prior = GMRF(μ, Q, LinearSolve.LDLtFactorization())

    # Poisson observation likelihood
    obs_model = ExponentialFamily(Poisson)
    obs_lik = obs_model(y)

    # Gaussian approximation
    posterior = gaussian_approximation(prior, obs_lik)

    # Evaluate log-density
    return logpdf(posterior, x_eval)
end

# Generate test data
println("\nGenerating test data...")
Random.seed!(123)

# True parameters: smooth mean field + moderate precision
μ_true = 0.5 .+ 0.3 .* sin.(range(0, 2π, length = n))
τ_true = 5.0
θ_true = vcat(μ_true, log(τ_true))

# Simulate observations (Poisson counts from smooth latent field)
x_true = μ_true .+ cumsum(randn(n)) .* sqrt(1 / τ_true) .* 0.5
x_true .-= mean(x_true)  # Center
y_obs = rand.(Poisson.(exp.(x_true .+ 0.5)))
x_eval = randn(n) .+ 0.3

# Initial parameter values (perturbed from truth)
θ_init = θ_true .+ randn(n_params) .* 0.1

println("  ✓ Generated $(length(y_obs)) Poisson observations")
println("  ✓ Initial parameters: $(n_params)-dimensional vector")

# Verify workflow works
println("\nVerifying workflow...")
f_val = benchmark_workflow(θ_init, y_obs, x_eval)
println("  ✓ Function value: $(@sprintf("%.4f", f_val))")

# Define backends
backends = [
    ("FiniteDiff", AutoFiniteDiff()),
    ("Zygote", AutoZygote()),
    ("Enzyme", AutoEnzyme(; function_annotation = Enzyme.Const)),
]

println("\n" * "="^80)
println("BENCHMARKING GRADIENT COMPUTATION")
println("="^80)

results = Dict()

for (name, backend) in backends
    println("\n$name:")
    println("-"^40)

    try
        # Warmup
        print("  Warming up... ")
        grad = DifferentiationInterface.gradient(
            θ -> benchmark_workflow(θ, y_obs, x_eval),
            backend,
            θ_init
        )
        println("✓")

        # Benchmark
        print("  Benchmarking... ")
        bench = @benchmark DifferentiationInterface.gradient(
            θ -> benchmark_workflow(θ, y_obs, x_eval),
            $backend,
            $θ_init
        ) samples = 10 seconds = 30

        results[name] = (
            gradient = grad,
            time = minimum(bench.times) / 1.0e6,  # Convert to ms
            bench = bench,
        )

        println("✓")
        println("  Time (min):     $(@sprintf("%.2f", results[name].time)) ms")
        println("  Time (median):  $(@sprintf("%.2f", median(bench.times) / 1.0e6)) ms")
        println("  Allocations:    $(bench.allocs)")
        println("  Memory:         $(@sprintf("%.2f", bench.memory / 1.0e6)) MB")

    catch e
        println("  ✗ Failed: $e")
        results[name] = nothing
    end
end

# Summary comparison
println("\n" * "="^80)
println("SUMMARY")
println("="^80)

if all(v !== nothing for v in values(results))
    # Verify gradients match
    println("\nGradient verification (comparing to FiniteDiff):")
    fd_grad = results["FiniteDiff"].gradient

    for name in ["Zygote", "Enzyme"]
        if results[name] !== nothing
            grad = results[name].gradient
            abs_error = abs.(grad - fd_grad)
            max_error = maximum(abs_error)
            mean_error = sum(abs_error) / length(abs_error)

            println("  $name:")
            println("    Max absolute error:  $(@sprintf("%.2e", max_error))")
            println("    Mean absolute error: $(@sprintf("%.2e", mean_error))")
        end
    end

    # Performance comparison table
    println("\nPerformance comparison:")
    println("  " * "─"^76)
    println(@sprintf("  %-20s %12s %12s %12s %12s", "Backend", "Time (ms)", "Speedup", "Allocs", "Memory (MB)"))
    println("  " * "─"^76)

    fd_time = results["FiniteDiff"].time
    for (name, backend) in backends
        if results[name] !== nothing
            r = results[name]
            speedup = fd_time / r.time
            speedup_str = name == "FiniteDiff" ? "1.0×" : @sprintf("%.1f×", speedup)

            println(
                @sprintf(
                    "  %-20s %12.2f %12s %12d %12.2f",
                    name, r.time, speedup_str, r.bench.allocs, r.bench.memory / 1.0e6
                )
            )
        end
    end
    println("  " * "─"^76)

    # Highlight winner
    if results["Enzyme"] !== nothing && results["Zygote"] !== nothing
        enzyme_vs_zygote = results["Zygote"].time / results["Enzyme"].time
        winner = enzyme_vs_zygote > 1 ? "Enzyme" : "Zygote"
        ratio = max(enzyme_vs_zygote, 1.0 / enzyme_vs_zygote)
        println("\n  🏆 $winner is fastest: $(@sprintf("%.1f", ratio))× faster than the other")
    end
end

println("\n" * "="^80)
println("BENCHMARK COMPLETE")
println("="^80)
