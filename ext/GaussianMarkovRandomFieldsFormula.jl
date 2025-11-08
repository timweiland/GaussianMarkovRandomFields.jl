module GaussianMarkovRandomFieldsFormula

using GaussianMarkovRandomFields
using StatsModels
using Distributions
using SparseArrays
using LinearAlgebra

import GaussianMarkovRandomFields:
    IID, RandomWalk, AR1, Besag, BYM2,
    RW1Model, AR1Model, IIDModel, BesagModel, BYM2Model, CombinedModel, FixedEffectsModel,
    LinearlyTransformedObservationModel, ExponentialFamily,
    BinomialObservations

include("GaussianMarkovRandomFieldsFormula/terms.jl")
include("GaussianMarkovRandomFieldsFormula/build.jl")

end
