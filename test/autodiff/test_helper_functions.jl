using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: _is_zero_tangent, _add_namedtuples
using ChainRulesCore
using Test

@testset "Autodiff Helper Functions" begin
    @testset "_is_zero_tangent" begin
        # Test with nothing
        @test _is_zero_tangent(nothing) == true

        # Test with NoTangent
        @test _is_zero_tangent(NoTangent()) == true

        # Test with ZeroTangent
        @test _is_zero_tangent(ZeroTangent()) == true

        # Test with regular values
        @test _is_zero_tangent(1.0) == false
        @test _is_zero_tangent([1.0, 2.0]) == false
        @test _is_zero_tangent((a = 1.0,)) == false
    end

    @testset "_add_namedtuples - Top level handling" begin
        # Both nothing
        @test _add_namedtuples(nothing, nothing) === nothing

        # One nothing, one NamedTuple
        nt1 = (a = 1.0, b = 2.0)
        @test _add_namedtuples(nothing, nt1) === nt1
        @test _add_namedtuples(nt1, nothing) === nt1

        # Both NoTangent
        @test _add_namedtuples(NoTangent(), NoTangent()) isa NoTangent

        # NoTangent with NamedTuple
        @test _add_namedtuples(NoTangent(), nt1) === nt1
        @test _add_namedtuples(nt1, NoTangent()) === nt1

        # NoTangent with nothing
        @test _add_namedtuples(NoTangent(), nothing) === nothing
        @test _add_namedtuples(nothing, NoTangent()) === nothing
    end

    @testset "_add_namedtuples - Key-wise addition" begin
        # Both non-nothing values
        nt1 = (a = 1.0, b = 2.0, c = [1.0, 2.0])
        nt2 = (a = 2.0, b = 3.0, c = [3.0, 4.0])
        result = _add_namedtuples(nt1, nt2)

        @test result.a ≈ 3.0
        @test result.b ≈ 5.0
        @test result.c ≈ [4.0, 6.0]

        # One value is nothing
        nt3 = (a = 1.0, b = nothing, c = [1.0, 2.0])
        nt4 = (a = 2.0, b = 3.0, c = [3.0, 4.0])
        result2 = _add_namedtuples(nt3, nt4)

        @test result2.a ≈ 3.0
        @test result2.b ≈ 3.0  # Takes the non-nothing value
        @test result2.c ≈ [4.0, 6.0]

        # Other value is nothing
        result3 = _add_namedtuples(nt4, nt3)
        @test result3.a ≈ 3.0
        @test result3.b ≈ 3.0  # Takes the non-nothing value
        @test result3.c ≈ [4.0, 6.0]

        # Both values are nothing for a key
        nt5 = (a = 1.0, b = nothing)
        nt6 = (a = 2.0, b = nothing)
        result4 = _add_namedtuples(nt5, nt6)

        @test result4.a ≈ 3.0
        @test result4.b === nothing
    end

    @testset "_add_namedtuples - Tangent types" begin
        # Create Tangent objects
        struct DummyType
            a::Float64
            b::Float64
        end

        t1 = Tangent{DummyType}(a = 1.0, b = 2.0)
        t2 = Tangent{DummyType}(a = 3.0, b = 4.0)

        # Tangent + NoTangent
        @test _add_namedtuples(t1, NoTangent()) === t1
        @test _add_namedtuples(NoTangent(), t1) === t1

        # Tangent + Tangent
        result = _add_namedtuples(t1, t2)
        @test result isa Tangent
        @test result.a ≈ 4.0
        @test result.b ≈ 6.0
    end

    @testset "_add_namedtuples - Complex nested case" begin
        # Test with nested structures
        nt1 = (
            mean = [1.0, 2.0],
            precision = nothing,
            extra = 3.0,
        )

        nt2 = (
            mean = [2.0, 3.0],
            precision = [4.0 0.0; 0.0 4.0],
            extra = nothing,
        )

        result = _add_namedtuples(nt1, nt2)

        @test result.mean ≈ [3.0, 5.0]
        @test result.precision ≈ [4.0 0.0; 0.0 4.0]  # Takes non-nothing value
        @test result.extra ≈ 3.0  # Takes non-nothing value
    end
end
