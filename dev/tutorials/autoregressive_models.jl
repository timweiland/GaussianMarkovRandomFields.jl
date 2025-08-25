using GaussianMarkovRandomFields, LinearAlgebra, SparseArrays

xs = 0:0.01:1
N = length(xs)
ϕ = 0.995
Λ₀ = 1.0e6
Λ = 1.0e3

μ = [ϕ^(i - 1) for i in eachindex(xs)]
main_diag = [[Λ₀]; repeat([Λ + ϕ^2], N - 2); [Λ]]
off_diag = repeat([-ϕ], N - 1)
Q = sparse(SymTridiagonal(main_diag, off_diag))
x = GMRF(μ, Q)

using Plots, Distributions
plot(xs, mean(x), ribbon = 1.96 * std(x), label = "Mean + std")
for i in 1:3
    plot!(xs, rand(x), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

using SparseArrays
A = spzeros(2, N)
A[1, 26] = 1.0
A[2, 76] = 1.0
y = [0.85, 0.71]
Λ_obs = 1.0e6
x_cond = condition_on_observations(x, A, Λ_obs, y)

plot(xs, mean(x_cond), ribbon = 1.96 * std(x_cond), label = "Mean + std")
for i in 1:3
    plot!(xs, rand(x_cond), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

W = spzeros(N, N)
for i in 1:N
    for k in [-2, -1, 1, 2]
        j = i + k
        if 1 <= j <= N
            W[i, j] = 1.0 / abs(k)
        end
    end
end

x_car = generate_car_model(W, 0.9; μ = μ, σ = 0.001)

plot(xs, mean(x_car), ribbon = 1.96 * std(x_car), label = "Mean + std")
for i in 1:3
    plot!(xs, rand(x_car), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

A = spzeros(3, N)
A[1, 1] = 1.0
A[2, 26] = 1.0
A[3, 76] = 1.0
y = [1.0, 0.85, 0.71]
Λ_obs = 1.0e6
x_car_cond = condition_on_observations(x_car, A, Λ_obs, y)
plot(xs, mean(x_car_cond), ribbon = 1.96 * std(x_car_cond), label = "Mean + std")
for i in 1:3
    plot!(xs, rand(x_car_cond), fillalpha = 0.3, linestyle = :dash, label = "Sample")
end
plot!()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
