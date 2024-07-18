using GMRFs, ReTest
include("GMRFsTests.jl")

if "skip-aqua" in ARGS
    GMRFTests.retest(r"\b(?!Aqua\b)\w+")
else
    GMRFTests.retest()
end
