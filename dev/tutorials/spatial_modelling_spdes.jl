meuse_path = joinpath(@__DIR__, "meuse.csv")
meuse_URL = "https://gist.githubusercontent.com/essicolo/91a2666f7c5972a91bca763daecdc5ff/raw/056bda04114d55b793469b2ab0097ec01a6d66c6/meuse.csv"
download(meuse_URL, meuse_path)

using CSV, DataFrames
df = DataFrame(CSV.File(meuse_path))
df[1:5, [:x, :y, :zinc]]

using Plots
x = convert(Vector{Float64}, df[:, :x])
y = convert(Vector{Float64}, df[:, :y])
zinc = df[:, :zinc]
scatter(x, y, zcolor = zinc)

using Random
train_idcs = randsubseq(1:size(df, 1), 0.85)
test_idcs = [i for i = 1:size(df, 1) if isempty(searchsorted(train_idcs, i))]
X = [x y]
X_train = X[train_idcs, :]
X_test = X[test_idcs, :]
y_train = zinc[train_idcs]
y_test = zinc[test_idcs]
size(X_train, 1), size(X_test, 1)

using GaussianMarkovRandomFields
points = zip(x, y)
grid = generate_mesh(points, 600.0, 100.0, save_path = "meuse.msh")

using Ferrite
ip = Lagrange{RefTriangle,1}()
qr = QuadratureRule{RefTriangle}(2)
disc = FEMDiscretization(grid, ip, qr)

spde = MaternSPDE{2}(range = 400.0, smoothness = 1)
u_matern = discretize(spde, disc)

Λ_obs = 10.0
A_train =
    evaluation_matrix(disc, [Tensors.Vec(X_train[i, :]...) for i = 1:size(X_train, 1)])
A_test = evaluation_matrix(disc, [Tensors.Vec(X_test[i, :]...) for i = 1:size(X_test, 1)])
u_cond = condition_on_observations(u_matern, A_train, Λ_obs, y_train)

rmse = (a, b) -> sqrt(mean((a .- b) .^ 2))
rmse(A_train * mean(u_cond), y_train), rmse(A_test * mean(u_cond), y_test)

VTKGridFile("meuse_mean", disc.dof_handler) do vtk
    write_solution(vtk, disc.dof_handler, mean(u_cond))
end
using Distributions
VTKGridFile("meuse_std", disc.dof_handler) do vtk
    write_solution(vtk, disc.dof_handler, std(u_cond))
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
