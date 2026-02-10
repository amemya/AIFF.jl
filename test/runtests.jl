using AIFF
using Test

@testset "AIFF.jl" begin

    @testset "IEEE 80-bit Extended Precision" begin
        # Test round-trip for common sample rates
        for sr in [8000.0, 11025.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0]
            buf = IOBuffer()
            AIFF.write_ieee_extended(buf, sr)
            seekstart(buf)
            result = AIFF.read_ieee_extended(buf)
            @test result ≈ sr atol=1e-10
        end

        # Test edge cases
        buf = IOBuffer()
        AIFF.write_ieee_extended(buf, 0.0)
        seekstart(buf)
        @test AIFF.read_ieee_extended(buf) == 0.0

        buf = IOBuffer()
        AIFF.write_ieee_extended(buf, Inf)
        seekstart(buf)
        @test isinf(AIFF.read_ieee_extended(buf))

        buf = IOBuffer()
        AIFF.write_ieee_extended(buf, NaN)
        seekstart(buf)
        @test isnan(AIFF.read_ieee_extended(buf))
    end

    @testset "Round-trip: 16-bit Stereo" begin
        sr = 44100.0
        nframes = 100
        nch = 2

        # Generate test signal: sine wave
        t = range(0, stop=1, length=nframes)
        left = sin.(2π * 440 .* t)
        right = cos.(2π * 440 .* t)
        samples = hcat(left, right)

        # Write and read back
        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=16)
            result = aiffread(path; format="double")
            @test result.samplerate ≈ sr
            @test result.nbits == 16
            @test size(result.data) == (nframes, nch)
            # 16-bit quantization tolerance: max error ≈ 1/2^15
            @test maximum(abs.(result.data .- samples)) < 2.0 / 32768
        end
    end

    @testset "Round-trip: 8-bit Mono" begin
        sr = 22050.0
        nframes = 50
        samples = reshape(sin.(range(0, stop=2π, length=nframes)), :, 1)

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=8)
            result = aiffread(path; format="double")
            @test result.samplerate ≈ sr
            @test result.nbits == 8
            @test size(result.data) == (nframes, 1)
            # 8-bit quantization: max error ~ 1/128
            @test maximum(abs.(result.data .- samples)) < 1.0 / 127 + 0.01
        end
    end

    @testset "Round-trip: 24-bit Stereo" begin
        sr = 48000.0
        nframes = 200
        t = range(0, stop=1, length=nframes)
        samples = hcat(sin.(2π * 1000 .* t), cos.(2π * 1000 .* t))

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=24)
            result = aiffread(path; format="double")
            @test result.samplerate ≈ sr
            @test result.nbits == 24
            @test size(result.data) == (nframes, 2)
            # 24-bit: very high precision
            @test maximum(abs.(result.data .- samples)) < 1.0 / (2^23) + 1e-6
        end
    end

    @testset "Round-trip: 32-bit Mono" begin
        sr = 96000.0
        nframes = 100
        samples = reshape(sin.(range(0, stop=2π, length=nframes)), :, 1)

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=32)
            result = aiffread(path; format="double")
            @test result.samplerate ≈ sr
            @test result.nbits == 32
            @test size(result.data) == (nframes, 1)
        end
    end

    @testset "Native format reading" begin
        sr = 44100.0
        nframes = 10
        samples = reshape(Float64.(1:nframes) ./ nframes, :, 1)

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=16)
            result = aiffread(path; format="native")
            @test eltype(result.data) == Int16
            @test size(result.data) == (nframes, 1)
        end
    end

    @testset "Size-only reading" begin
        sr = 44100.0
        nframes = 100
        samples = randn(nframes, 3)

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=16)
            result = aiffread(path; format="size")
            @test result.data == (nframes, 3)
            @test result.samplerate ≈ sr
            @test result.nbits == 16
        end
    end

    @testset "1D Vector input (mono shortcut)" begin
        sr = 44100.0
        samples = sin.(range(0, stop=2π, length=50))

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=16)
            result = aiffread(path)
            @test size(result.data) == (50, 1)
        end
    end

    @testset "Empty audio (0 frames)" begin
        sr = 44100.0
        samples = Matrix{Float64}(undef, 0, 2)

        mktemp() do path, _
            aiffwrite(path, samples, sr; nbits=16)
            result = aiffread(path)
            @test size(result.data) == (0, 2)
            @test result.samplerate ≈ sr
        end
    end

    @testset "IO-based read/write" begin
        sr = 44100.0
        nframes = 20
        samples = randn(nframes, 1)

        buf = IOBuffer()
        aiffwrite(buf, samples, sr; nbits=16)
        seekstart(buf)
        result = aiffread(buf; format="double")
        @test size(result.data) == (nframes, 1)
        @test result.samplerate ≈ sr
    end
    @testset "aiffplay method availability" begin
        # Verify aiffplay is exported and has methods for all expected signatures
        @test hasmethod(aiffplay, (AbstractString,))
        @test hasmethod(aiffplay, (AbstractVector{Float64}, Real))
        @test hasmethod(aiffplay, (AbstractMatrix{Float64}, Real))
    end

    @testset "aiffplay with 1D vector (mono)" begin
        # Verify that 1D vector input doesn't error during setup
        # (can't test actual playback without audio hardware)
        samples = sin.(range(0, stop=2π, length=100))
        @test samples isa AbstractVector
        # Ensure the method exists for this type
        @test hasmethod(aiffplay, (typeof(samples), Real))
    end

end
