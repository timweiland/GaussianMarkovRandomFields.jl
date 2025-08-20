#!/usr/bin/env sh
# Usage example: ./test_only.sh "Gaussian Approximation"
# ^ will run only testsets that contain the string "Gaussian Approximation"
#   in their description

julia --project -e "using TestEnv; TestEnv.activate(); include(\"test/GaussianMarkovRandomFieldsTests.jl\"); GaussianMarkovRandomFieldsTests.retest(\"$1\")"
