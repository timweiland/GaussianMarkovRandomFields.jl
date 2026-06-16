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

using Zygote, Enzyme, FiniteDiff, Mooncake

using CliqueTrees.Multifrontal: symbolic, chordal

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
function benchmark_workflow(θ::Vector{Float64}, y::PoissonObservations, x_eval::Vector{Float64})
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

# ChordalGMRF workflow (only supports Mooncake)
function benchmark_workflow_chordal(θ::Vector{Float64}, y::PoissonObservations, x_eval::Vector{Float64})
    # Extract hyperparameters
    μ = θ[1:n]      # Mean field (100 params)
    log_τ = θ[n + 1]  # Log precision
    τ = exp(log_τ)

    # Build precision matrix using RW1 latent model
    rw1_model = RW1Model(n)
    Q = sparse(precision_matrix(rw1_model; τ = τ))

    # Prior ChordalGMRF with custom mean
    prior = ChordalGMRF(μ, Q)

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
y_counts = rand.(Poisson.(exp.(x_true .+ 0.5)))
y_obs = PoissonObservations(y_counts)
x_eval = randn(n) .+ 0.3

# Initial parameter values (perturbed from truth)
θ_init = θ_true .+ randn(n_params) .* 0.1

println("  ✓ Generated $(length(y_counts)) Poisson observations")
println("  ✓ Initial parameters: $(n_params)-dimensional vector")

# Verify workflows work
println("\nVerifying workflows...")
f_val = benchmark_workflow(θ_init, y_obs, x_eval)
println("  ✓ GMRF function value: $(@sprintf("%.4f", f_val))")

f_val_chordal = benchmark_workflow_chordal(θ_init, y_obs, x_eval)
println("  ✓ ChordalGMRF function value: $(@sprintf("%.4f", f_val_chordal))")
println("  ✓ Difference: $(@sprintf("%.2e", abs(f_val - f_val_chordal)))")

# Define backends
backends = [
    ("FiniteDiff", AutoFiniteDiff()),
    ("Zygote", AutoZygote()),
    ("Enzyme", AutoEnzyme(; function_annotation = Enzyme.Const)),
]

println("\n" * "="^80)
println("BENCHMARKING GRADIENT COMPUTATION (via DifferentiationInterface, prepared)")
println("="^80)

results = Dict()

for (name, backend) in backends
    println("\n$name:")
    println("-"^40)

    try
        # Define loss function
        loss = θ -> benchmark_workflow(θ, y_obs, x_eval)

        # Prepare gradient (includes warmup/compilation)
        print("  Preparing... ")
        prep = DifferentiationInterface.prepare_gradient(loss, backend, θ_init)
        println("✓")

        # Compute gradient once
        print("  Computing gradient... ")
        grad = DifferentiationInterface.gradient(loss, prep, backend, θ_init)
        println("✓")

        # Benchmark with prepared gradient
        print("  Benchmarking... ")
        bench = @benchmark DifferentiationInterface.gradient($loss, $prep, $backend, $θ_init) samples = 10 seconds = 30

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
        println("  ✗ Failed: $(typeof(e).name.name)")
        if e isa ErrorException
            println("    $(first(split(e.msg, '\n')))")
        end
        results[name] = nothing
    end
end

# ChordalGMRF benchmark (Mooncake only)
println("\n" * "="^80)
println("BENCHMARKING ChordalGMRF (Mooncake only)")
println("="^80)

println("\nChordalGMRF + Mooncake:")
println("-"^40)

try
    # Define loss function
    loss_chordal = θ -> benchmark_workflow_chordal(θ, y_obs, x_eval)

    # Prepare gradient (includes warmup/compilation)
    print("  Preparing... ")
    prep_chordal = DifferentiationInterface.prepare_gradient(loss_chordal, AutoMooncake(; config = nothing), θ_init)
    println("✓")

    # Compute gradient once
    print("  Computing gradient... ")
    grad_chordal = DifferentiationInterface.gradient(loss_chordal, prep_chordal, AutoMooncake(; config = nothing), θ_init)
    println("✓")

    # Benchmark with prepared gradient
    print("  Benchmarking... ")
    bench_chordal = @benchmark DifferentiationInterface.gradient($loss_chordal, $prep_chordal, AutoMooncake(; config = nothing), $θ_init) samples = 10 seconds = 30

    results["ChordalGMRF+Mooncake"] = (
        gradient = grad_chordal,
        time = minimum(bench_chordal.times) / 1.0e6,
        bench = bench_chordal,
    )

    println("✓")
    println("  Time (min):     $(@sprintf("%.2f", results["ChordalGMRF+Mooncake"].time)) ms")
    println("  Time (median):  $(@sprintf("%.2f", median(bench_chordal.times) / 1.0e6)) ms")
    println("  Allocations:    $(bench_chordal.allocs)")
    println("  Memory:         $(@sprintf("%.2f", bench_chordal.memory / 1.0e6)) MB")

catch e
    println("  ✗ Failed: $(typeof(e).name.name)")
    if e isa ErrorException
        println("    $(first(split(e.msg, '\n')))")
    end
    results["ChordalGMRF+Mooncake"] = nothing
end

# Summary comparison
println("\n" * "="^80)
println("SUMMARY")
println("="^80)

if results["FiniteDiff"] !== nothing
    # Verify gradients match
    println("\nGradient verification (comparing to FiniteDiff):")
    fd_grad = results["FiniteDiff"].gradient

    for name in ["Zygote", "Enzyme", "ChordalGMRF+Mooncake"]
        if get(results, name, nothing) !== nothing
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
    println("\nPerformance comparison (GMRF backends):")
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
        println("\n  GMRF winner: $winner ($(@sprintf("%.1f", ratio))× faster)")
    end
end

# ChordalGMRF vs GMRF comparison (Zygote only)
if get(results, "ChordalGMRF+Mooncake", nothing) !== nothing && get(results, "Zygote", nothing) !== nothing
    println("\n" * "="^80)
    println("GMRF vs ChordalGMRF COMPARISON (Zygote vs Mooncake)")
    println("="^80)

    r_gmrf = results["Zygote"]
    r_chordal = results["ChordalGMRF+Mooncake"]

    println("\n  " * "─"^76)
    println(@sprintf("  %-20s %12s %12s %12s %12s", "Implementation", "Time (ms)", "Speedup", "Allocs", "Memory (MB)"))
    println("  " * "─"^76)

    println(
        @sprintf(
            "  %-20s %12.2f %12s %12d %12.2f",
            "GMRF", r_gmrf.time, "1.0×", r_gmrf.bench.allocs, r_gmrf.bench.memory / 1.0e6
        )
    )

    chordal_speedup = r_gmrf.time / r_chordal.time
    println(
        @sprintf(
            "  %-20s %12.2f %12s %12d %12.2f",
            "ChordalGMRF", r_chordal.time, @sprintf("%.1f×", chordal_speedup),
            r_chordal.bench.allocs, r_chordal.bench.memory / 1.0e6
        )
    )

    println("  " * "─"^76)

    winner = chordal_speedup > 1 ? "ChordalGMRF" : "GMRF"
    ratio = max(chordal_speedup, 1.0 / chordal_speedup)
    println("\n  Winner: $winner ($(@sprintf("%.1f", ratio))× faster)")
end

println("\n" * "="^80)
println("BENCHMARK COMPLETE")
println("="^80)
