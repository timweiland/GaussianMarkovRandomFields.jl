using GaussianMarkovRandomFields, ReTest
include("GaussianMarkovRandomFieldsTests.jl")

if "skip-aqua" in ARGS
    GaussianMarkovRandomFieldsTests.retest(r"\b(?!Aqua\b)\w+")
else
    GaussianMarkovRandomFieldsTests.retest()
end
