#!/usr/bin/env julia
"""
Compare two benchmark JSON files (PR vs baseline) and emit a Markdown table.

Usage:
    julia --project=. compare_results.jl <baseline.json> <pr.json> [<output.md>]

The script uses `BenchmarkTools.judge` on the median time of each leaf
benchmark. Any benchmark judged as a regression at the default threshold is
flagged. The script exits with a non-zero code if any regression is detected.
"""

using BenchmarkTools

if length(ARGS) < 2
    error("Usage: julia compare_results.jl <baseline.json> <pr.json> [<output.md>]")
end

baseline_path = ARGS[1]
pr_path = ARGS[2]
output_path = length(ARGS) >= 3 ? ARGS[3] : nothing

baseline = BenchmarkTools.load(baseline_path)[1]::BenchmarkGroup
pr = BenchmarkTools.load(pr_path)[1]::BenchmarkGroup

# Regression threshold: how much slower the PR can be before we flag it.
# CI runners are noisy, so use a generous threshold by default.
const REGRESSION_TOLERANCE = 0.5  # 50% slowdown triggers a regression flag
const IMPROVEMENT_TOLERANCE = 0.2 # 20% speedup highlighted as improvement

judgement = judge(median(pr), median(baseline); time_tolerance = REGRESSION_TOLERANCE)

regressions_list = BenchmarkTools.leaves(regressions(judgement))
improvements_list = BenchmarkTools.leaves(improvements(judgement))

function fmt_time(ns)
    return BenchmarkTools.prettytime(ns)
end

io = IOBuffer()

println(io, "## Performance benchmark comparison")
println(io)
println(io, "Comparing PR (`", pr_path, "`) against baseline (`", baseline_path, "`).")
println(io, "Regression tolerance: ", round(Int, REGRESSION_TOLERANCE * 100), "% slowdown.")
println(io)

if isempty(regressions_list)
    println(io, "**No regressions detected.**")
else
    println(io, "### ⚠️ Regressions (", length(regressions_list), ")")
    println(io)
    println(io, "| Benchmark | Baseline | PR | Ratio |")
    println(io, "|---|---:|---:|---:|")
    for (keys, trial) in regressions_list
        name = join(keys, " / ")
        b_time = time(median(baseline[keys...]))
        p_time = time(median(pr[keys...]))
        ratio = p_time / b_time
        println(
            io, "| `", name, "` | ", fmt_time(b_time), " | ", fmt_time(p_time),
            " | **", round(ratio; digits = 2), "x** |"
        )
    end
    println(io)
end

if !isempty(improvements_list)
    println(io, "### ✅ Improvements (", length(improvements_list), ")")
    println(io)
    println(io, "| Benchmark | Baseline | PR | Ratio |")
    println(io, "|---|---:|---:|---:|")
    for (keys, trial) in improvements_list
        name = join(keys, " / ")
        b_time = time(median(baseline[keys...]))
        p_time = time(median(pr[keys...]))
        ratio = p_time / b_time
        println(
            io, "| `", name, "` | ", fmt_time(b_time), " | ", fmt_time(p_time),
            " | ", round(ratio; digits = 2), "x |"
        )
    end
    println(io)
end

function _safe_get(group, keys)
    g = group
    for k in keys
        g isa BenchmarkGroup || return nothing
        haskey(g, k) || return nothing
        g = g[k]
    end
    return g
end

# Always include the full table for visibility.
println(io, "<details>")
println(io, "<summary>Full results</summary>")
println(io)
println(io, "| Benchmark | Baseline | PR | Ratio |")
println(io, "|---|---:|---:|---:|")
for (keys, trial) in BenchmarkTools.leaves(pr)
    name = join(keys, " / ")
    p_time = time(median(trial))
    baseline_trial = _safe_get(baseline, keys)
    if baseline_trial !== nothing
        b_time = time(median(baseline_trial))
        ratio = p_time / b_time
        println(
            io, "| `", name, "` | ", fmt_time(b_time), " | ", fmt_time(p_time),
            " | ", round(ratio; digits = 2), "x |"
        )
    else
        println(io, "| `", name, "` | _new_ | ", fmt_time(p_time), " | — |")
    end
end
println(io, "</details>")

markdown = String(take!(io))
println(markdown)

if output_path !== nothing
    open(output_path, "w") do io
        write(io, markdown)
    end
end

if !isempty(regressions_list)
    @warn "Performance regressions detected"
    exit(1)
end
