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

# Work on the median estimate trees: `judge` compares these, so the keys of its
# leaves index exactly into them. (Re-indexing the *raw* groups with those keys
# can land on a sub-group when the two suites differ in shape, and
# `time(median(group))` returns a BenchmarkGroup rather than a number.)
med_baseline = median(baseline)
med_pr = median(pr)

judgement = judge(med_pr, med_baseline; time_tolerance = REGRESSION_TOLERANCE)

# Flag on the median *time* only. `regressions`/`improvements` also fire on memory, but the
# default memory tolerance is ~1% — far too sensitive for noisy CI runners, where a
# sub-percent allocation wobble would fail the job even when time is unchanged or improved.
_judged_leaves = BenchmarkTools.leaves(judgement)
regressions_list = [(keys, j) for (keys, j) in _judged_leaves if time(j) === :regression]
improvements_list = [(keys, j) for (keys, j) in _judged_leaves if time(j) === :improvement]

fmt_time(ns) = BenchmarkTools.prettytime(ns)

# Walk a (possibly nested) BenchmarkGroup along `keys`; return the leaf or `nothing`
# if the path is missing or stops at a sub-group rather than a leaf estimate.
function _safe_get(group, keys)
    g = group
    for k in keys
        g isa BenchmarkGroup || return nothing
        haskey(g, k) || return nothing
        g = g[k]
    end
    return g
end

# Median time (ns) of the leaf at `keys` in a median estimate tree, or `nothing`.
function _leaf_time(med_group, keys)
    g = _safe_get(med_group, keys)
    return g isa BenchmarkTools.TrialEstimate ? time(g) : nothing
end

# Emit a table for the given leaf list; returns the number of rows actually written.
function _emit_rows!(io, leaf_list; bold_ratio::Bool)
    rows = 0
    for (keys, _judgement) in leaf_list
        b_time = _leaf_time(med_baseline, keys)
        p_time = _leaf_time(med_pr, keys)
        (b_time === nothing || p_time === nothing) && continue
        name = join(keys, " / ")
        ratio = p_time / b_time
        ratio_str = bold_ratio ? string("**", round(ratio; digits = 2), "x**") :
            string(round(ratio; digits = 2), "x")
        println(io, "| `", name, "` | ", fmt_time(b_time), " | ", fmt_time(p_time), " | ", ratio_str, " |")
        rows += 1
    end
    return rows
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
    _emit_rows!(io, regressions_list; bold_ratio = true)
    println(io)
end

if !isempty(improvements_list)
    println(io, "### ✅ Improvements (", length(improvements_list), ")")
    println(io)
    println(io, "| Benchmark | Baseline | PR | Ratio |")
    println(io, "|---|---:|---:|---:|")
    _emit_rows!(io, improvements_list; bold_ratio = false)
    println(io)
end

# Always include the full table for visibility.
println(io, "<details>")
println(io, "<summary>Full results</summary>")
println(io)
println(io, "| Benchmark | Baseline | PR | Ratio |")
println(io, "|---|---:|---:|---:|")
for (keys, est) in BenchmarkTools.leaves(med_pr)
    name = join(keys, " / ")
    p_time = time(est)
    b_time = _leaf_time(med_baseline, keys)
    if b_time !== nothing
        ratio = p_time / b_time
        println(io, "| `", name, "` | ", fmt_time(b_time), " | ", fmt_time(p_time), " | ", round(ratio; digits = 2), "x |")
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
