# Linux PulseAudio playback via libpulse-simple
module AIFFPlay
import ..aiffplay

import Libdl

# ============================================================================
# PulseAudio Constants & Types
# ============================================================================

# pa_sample_format enum
const PA_SAMPLE_U8        =  0  # Unsigned 8 Bit PCM
const PA_SAMPLE_ALAW      =  1  # 8 Bit a-Law
const PA_SAMPLE_ULAW      =  2  # 8 Bit mu-Law
const PA_SAMPLE_S16LE     =  3  # Signed 16 Bit PCM, little endian
const PA_SAMPLE_S16BE     =  4  # Signed 16 Bit PCM, big endian
const PA_SAMPLE_FLOAT32LE =  5  # 32 Bit IEEE float, little endian
const PA_SAMPLE_FLOAT32BE =  6  # 32 Bit IEEE float, big endian
const PA_SAMPLE_S32LE     =  7  # Signed 32 Bit PCM, little endian
const PA_SAMPLE_S32BE     =  8  # Signed 32 Bit PCM, big endian
const PA_SAMPLE_S24LE     =  9  # Signed 24 Bit PCM packed, little endian
const PA_SAMPLE_S24BE     = 10  # Signed 24 Bit PCM packed, big endian
const PA_SAMPLE_S24_32LE  = 11  # Signed 24 Bit PCM in LSB of 32 Bit words, LE
const PA_SAMPLE_S24_32BE  = 12  # Signed 24 Bit PCM in LSB of 32 Bit words, BE

const PA_STREAM_PLAYBACK = 1

struct pa_sample_spec
    format::Int32
    rate::UInt32
    channels::UInt8
end

struct pa_buffer_attr
    maxlength::UInt32
    tlength::UInt32
    prebuf::UInt32
    minreq::UInt32
    fragsize::UInt32
end

# pa_channel_map with 32 channel slots
struct pa_channel_map
    channels::UInt8
    map0::Cint;  map1::Cint;  map2::Cint;  map3::Cint
    map4::Cint;  map5::Cint;  map6::Cint;  map7::Cint
    map8::Cint;  map9::Cint;  map10::Cint; map11::Cint
    map12::Cint; map13::Cint; map14::Cint; map15::Cint
    map16::Cint; map17::Cint; map18::Cint; map19::Cint
    map20::Cint; map21::Cint; map22::Cint; map23::Cint
    map24::Cint; map25::Cint; map26::Cint; map27::Cint
    map28::Cint; map29::Cint; map30::Cint; map31::Cint

    pa_channel_map() = new(0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
end

const pa_simple = Ptr{Cvoid}
const LibPulseSimple = Libdl.find_library(["libpulse-simple", "libpulse-simple.so.0"])

# ============================================================================
# Playback
# ============================================================================

function aiffplay(data::AbstractVecOrMat{<:Real}, fs::Real)
    if LibPulseSimple == ""
        error("libpulse-simple not found. Install your system's PulseAudio or pipewire-pulse compatibility package providing libpulse-simple.so.0 (e.g., on Debian/Ubuntu: sudo apt install libpulse0).")
    end

    samples = ndims(data) == 1 ? reshape(data, :, 1) : data
    nChannels = size(samples, 2)
    ss = pa_sample_spec(PA_SAMPLE_FLOAT32LE, round(UInt32, fs), UInt8(nChannels))

    # Interleave samples as Float32 (PulseAudio expects interleaved)
    # Normalize integers to [-1.0, 1.0] range
    buf = Vector{Float32}(undef, size(samples, 1) * nChannels)
    idx = 1
    if eltype(samples) <: Integer
        maxval = Float32(typemax(eltype(samples)))
        for i in 1:size(samples, 1)
            for j in 1:nChannels
                buf[idx] = Float32(samples[i, j]) / maxval
                idx += 1
            end
        end
    else
        for i in 1:size(samples, 1)
            for j in 1:nChannels
                buf[idx] = Float32(samples[i, j])
                idx += 1
            end
        end
    end

    # Open connection
    s = ccall((:pa_simple_new, LibPulseSimple),
              pa_simple,
              (Cstring, Cstring, Cint, Cstring, Cstring,
               Ptr{pa_sample_spec}, Ptr{pa_channel_map},
               Ptr{pa_buffer_attr}, Ptr{Cint}),
              C_NULL,              # default server
              "Julia AIFF.jl",     # application name
              PA_STREAM_PLAYBACK,
              C_NULL,              # default device
              "aiffplay",          # stream description
              Ref(ss),
              C_NULL,              # default channel map
              C_NULL,              # default buffer attributes
              C_NULL)              # ignore error code
    if s == C_NULL
        error("pa_simple_new failed")
    end

    try
        # Write audio data
        write_ret = ccall((:pa_simple_write, LibPulseSimple),
                          Cint,
                          (pa_simple, Ptr{Cvoid}, Csize_t, Ptr{Cint}),
                          s, buf, sizeof(buf), C_NULL)
        write_ret != 0 && error("pa_simple_write failed with $write_ret")

        # Wait for playback to complete
        drain_ret = ccall((:pa_simple_drain, LibPulseSimple),
                          Cint,
                          (pa_simple, Ptr{Cint}),
                          s, C_NULL)
        drain_ret != 0 && error("pa_simple_drain failed with $drain_ret")
    finally
        ccall((:pa_simple_free, LibPulseSimple), Cvoid, (pa_simple,), s)
    end
end

end # module AIFFPlay
