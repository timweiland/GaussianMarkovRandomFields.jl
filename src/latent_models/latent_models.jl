include("latent_model.jl")
include("local_quadratic.jl")
include("utils.jl")
include("ar.jl")
include("rw.jl")
include("iid.jl")
include("besag.jl")
include("bym2.jl")
include("combined.jl")
include("separable.jl")
include("fixed_effects.jl")
# NOTE: latent_models/gaussian_approximation.jl is included from the top-level
# module *after* observation_models and the arithmetic/workspace Newton loops,
# since its method signatures reference `ObservationLikelihood` and its bodies
# call `_newton_loop` / `_workspace_newton_loop`.
