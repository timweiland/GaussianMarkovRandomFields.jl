@testset "CompositeObservations" begin
    @testset "Constructor and basic properties" begin
        # Test basic construction
        y1 = [1.0, 2.0, 3.0]
        y2 = [4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        @test length(y_composite) == 5
        @test eltype(y_composite) == Float64
        @test y_composite isa AbstractVector{Float64}
    end

    @testset "AbstractVector interface" begin
        y1 = [1.0, 2.0, 3.0]
        y2 = [4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        # Test indexing
        @test y_composite[1] == 1.0
        @test y_composite[2] == 2.0
        @test y_composite[3] == 3.0
        @test y_composite[4] == 4.0
        @test y_composite[5] == 5.0

        # Test iteration
        collected = collect(y_composite)
        @test collected == [1.0, 2.0, 3.0, 4.0, 5.0]

        # Test size
        @test size(y_composite) == (5,)
    end

    @testset "Edge cases" begin
        # Single component
        y_single = CompositeObservations(([1.0, 2.0],))
        @test length(y_single) == 2
        @test y_single[1] == 1.0

        # Empty components should error
        @test_throws ArgumentError CompositeObservations(())

        # Mixed numeric types should convert to Float64
        y_mixed = CompositeObservations(([1, 2], [3.0, 4.0]))
        @test eltype(y_mixed) == Float64
        @test y_mixed[1] == 1.0
    end

    @testset "Component access" begin
        y1 = [1.0, 2.0]
        y2 = [3.0, 4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        # Should be able to access components
        @test length(y_composite.components) == 2
        @test y_composite.components[1] == y1
        @test y_composite.components[2] == y2
    end

    @testset "Iterator component transitions" begin
        # Test iterator transitions between multiple components
        y1 = [1.0]
        y2 = [2.0]
        y3 = [3.0, 4.0]
        y_composite = CompositeObservations((y1, y2, y3))

        # Collect should iterate through all components correctly
        collected = collect(y_composite)
        @test collected == [1.0, 2.0, 3.0, 4.0]

        # Test manual iteration to trigger component transitions
        iter_state = iterate(y_composite)
        @test iter_state[1] == 1.0  # First element

        iter_state = iterate(y_composite, iter_state[2])
        @test iter_state[1] == 2.0  # Second element (different component)

        iter_state = iterate(y_composite, iter_state[2])
        @test iter_state[1] == 3.0  # Third element (different component)
    end

    @testset "Index boundary cases" begin
        # Test indexing at component boundaries
        y1 = [1.0, 2.0]
        y2 = [3.0]
        y3 = [4.0, 5.0]
        y_composite = CompositeObservations((y1, y2, y3))

        # Test accessing elements at component boundaries
        @test y_composite[2] == 2.0  # Last element of first component
        @test y_composite[3] == 3.0  # Single element in second component
        @test y_composite[4] == 4.0  # First element of third component

        # Test bounds checking
        @test_throws BoundsError y_composite[0]
        @test_throws BoundsError y_composite[6]  # Past the end
    end
end
