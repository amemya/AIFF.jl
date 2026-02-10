module AIFF

export aiffread, aiffwrite

# ============================================================================
# Data Types
# ============================================================================

"""
    AIFFChunk

Generic container for unknown/unrecognized chunks, preserved for round-trip fidelity.
"""
struct AIFFChunk
    id::Symbol
    data::Vector{UInt8}
end

"""
    AIFFFormat

Parsed representation of the Common Chunk ('COMM').
"""
struct AIFFFormat
    nchannels::Int16
    nframes::UInt32
    samplebits::Int16
    samplerate::Float64
end

"""
    AIFFMarker

A marker pointing to a position in the sound data.
"""
struct AIFFMarker
    id::Int16
    position::UInt32
    name::String
end

"""
    AIFFLoop

Loop point specification (used by Instrument Chunk).
"""
struct AIFFLoop
    playmode::Int16
    beginloop::Int16
    endloop::Int16
end

"""
    AIFFInstrument

Instrument parameters from the 'INST' chunk.
"""
struct AIFFInstrument
    basenote::Int8
    detune::Int8
    lownote::Int8
    highnote::Int8
    lowvelocity::Int8
    highvelocity::Int8
    gain::Int16
    sustainloop::AIFFLoop
    releaseloop::AIFFLoop
end

# ============================================================================
# Big-Endian I/O Helpers
# ============================================================================

"""
    read_be(io, T) -> T

Read a value of type `T` from `io` in big-endian byte order.
"""
function read_be(io::IO, ::Type{T}) where {T<:Union{Int8,UInt8}}
    return read(io, T)
end

function read_be(io::IO, ::Type{T}) where {T<:Union{Int16,UInt16,Int32,UInt32,Int64,UInt64}}
    return ntoh(read(io, T))
end

"""
    write_be(io, x) -> Int

Write a value `x` to `io` in big-endian byte order. Returns bytes written.
"""
function write_be(io::IO, x::T) where {T<:Union{Int8,UInt8}}
    return write(io, x)
end

function write_be(io::IO, x::T) where {T<:Union{Int16,UInt16,Int32,UInt32,Int64,UInt64}}
    return write(io, hton(x))
end

"""
    read_chunkid(io) -> Symbol

Read a 4-byte chunk ID and return it as a Symbol.
"""
function read_chunkid(io::IO)
    bytes = read(io, 4)
    length(bytes) == 4 || error("Unexpected end of file while reading chunk ID")
    return Symbol(String(bytes))
end

"""
    write_chunkid(io, id::Symbol) -> Int

Write a 4-byte chunk ID from a Symbol.
"""
function write_chunkid(io::IO, id::Symbol)
    s = String(id)
    length(s) == 4 || error("Chunk ID must be exactly 4 characters: got '$(s)'")
    return write(io, codeunits(s))
end

# ============================================================================
# IEEE 80-bit Extended Precision Conversion
# ============================================================================

"""
    read_ieee_extended(io) -> Float64

Read an 80-bit IEEE 754 extended precision float (big-endian) from `io` and
convert it to `Float64`. Used for the sample rate field in AIFF Common Chunks.

Layout (10 bytes, big-endian):
  - Bit  79:      sign (1 bit)
  - Bits 78–64:   exponent (15 bits, bias 16383)
  - Bits 63–0:    mantissa (64 bits, explicit integer bit)
"""
function read_ieee_extended(io::IO)
    bytes = read(io, 10)
    length(bytes) == 10 || error("Unexpected end of file while reading IEEE extended")

    # Extract sign (bit 79)
    sign = (bytes[1] >> 7) & 0x01

    # Extract exponent (bits 78–64, 15 bits)
    exponent = (UInt16(bytes[1] & 0x7F) << 8) | UInt16(bytes[2])

    # Extract mantissa (bits 63–0, 64 bits)
    mantissa = UInt64(0)
    for i in 3:10
        mantissa = (mantissa << 8) | UInt64(bytes[i])
    end

    # Special cases
    if exponent == 0 && mantissa == 0
        return 0.0
    elseif exponent == 0x7FFF  # all 1s
        if mantissa == 0
            return sign == 1 ? -Inf : Inf
        else
            return NaN
        end
    end

    # Normal case: value = (-1)^sign * mantissa * 2^(exponent - 16383 - 63)
    # The mantissa includes the explicit integer bit (bit 63)
    f = ldexp(Float64(mantissa), Int(exponent) - 16383 - 63)
    return sign == 1 ? -f : f
end

"""
    write_ieee_extended(io, x::Float64)

Write a `Float64` value as an 80-bit IEEE 754 extended precision float
(big-endian) to `io`.
"""
function write_ieee_extended(io::IO, x::Float64)
    bytes = zeros(UInt8, 10)

    if x == 0.0
        write(io, bytes)
        return 10
    end

    sign = UInt8(0)
    if x < 0
        sign = UInt8(1)
        x = -x
    end

    if isinf(x)
        bytes[1] = (sign << 7) | 0x7F
        bytes[2] = 0xFF
        # mantissa = 0 for infinity (bytes 3-10 already zero)
        write(io, bytes)
        return 10
    end

    if isnan(x)
        bytes[1] = 0x7F
        bytes[2] = 0xFF
        bytes[3] = 0xC0  # quiet NaN
        write(io, bytes)
        return 10
    end

    # Decompose the Float64
    # frexp returns (fraction, exponent) where fraction ∈ [0.5, 1.0)
    # We need mantissa with explicit integer bit set (bit 63)
    frac, exp = frexp(x)

    # Convert to integer mantissa: shift fraction by 64 bits
    # frac ∈ [0.5, 1.0), so frac * 2^64 ∈ [2^63, 2^64)
    mantissa = unsafe_trunc(UInt64, ldexp(frac, 64))

    # Exponent for 80-bit: bias is 16383. frexp gives exp such that
    # x = frac * 2^exp, and mantissa = frac * 2^64
    # So x = mantissa * 2^(exp - 64)
    # In 80-bit format: x = mantissa * 2^(biased_exp - 16383 - 63)
    # Therefore: exp - 64 = biased_exp - 16383 - 63
    #           biased_exp = exp - 64 + 16383 + 63 = exp + 16382
    biased_exp = exp + 16382

    bytes[1] = (sign << 7) | UInt8((biased_exp >> 8) & 0x7F)
    bytes[2] = UInt8(biased_exp & 0xFF)

    for i in 3:10
        bytes[i] = UInt8((mantissa >> (8 * (10 - i))) & 0xFF)
    end

    write(io, bytes)
    return 10
end

# ============================================================================
# PCM Data Conversion
# ============================================================================

"""
    bytes_per_sample(nbits) -> Int

Number of bytes used to store one sample point of `nbits` bit depth.
"""
bytes_per_sample(nbits::Integer) = cld(nbits, 8)

"""
    read_pcm_samples(io, format::AIFFFormat, offset::UInt32) -> Matrix

Read PCM sample data from the SSND chunk and return as a Matrix{T}.
Rows = sample frames, Columns = channels.
"""
function read_pcm_samples(io::IO, fmt::AIFFFormat, offset::UInt32)
    # Skip offset bytes
    if offset > 0
        skip(io, offset)
    end

    nbits = Int(fmt.samplebits)
    nch = Int(fmt.nchannels)
    nframes = Int(fmt.nframes)
    bps = bytes_per_sample(nbits)

    # Determine the native integer type for storage
    T = if bps == 1
        Int8
    elseif bps == 2
        Int16
    elseif bps == 3
        Int32  # stored in 3 bytes, we'll expand to Int32
    elseif bps == 4
        Int32
    else
        error("Unsupported bit depth: $nbits")
    end

    samples = Matrix{T}(undef, nframes, nch)

    for i in 1:nframes
        for ch in 1:nch
            if bps == 1
                samples[i, ch] = read_be(io, Int8)
            elseif bps == 2
                samples[i, ch] = read_be(io, Int16)
            elseif bps == 3
                # Read 3 bytes big-endian, sign-extend to Int32
                b1 = read(io, UInt8)
                b2 = read(io, UInt8)
                b3 = read(io, UInt8)
                raw = (Int32(b1) << 24) | (Int32(b2) << 16) | (Int32(b3) << 8)
                samples[i, ch] = raw >> 8  # arithmetic shift to sign-extend
            elseif bps == 4
                samples[i, ch] = read_be(io, Int32)
            end
        end
    end

    return samples
end

"""
    pcm_to_float(samples::Matrix, nbits::Integer) -> Matrix{Float64}

Convert integer PCM samples to Float64 in the range [-1.0, 1.0].
"""
function pcm_to_float(samples::Matrix{T}, nbits::Integer) where {T<:Integer}
    maxval = Float64(1 << (nbits - 1))
    return Float64.(samples) ./ maxval
end

"""
    float_to_pcm(samples::Matrix{Float64}, nbits::Integer) -> Matrix

Convert Float64 samples ([-1.0, 1.0]) to integer PCM values.
"""
function float_to_pcm(samples::Matrix{Float64}, nbits::Integer)
    bps = bytes_per_sample(nbits)
    maxval = Float64((1 << (nbits - 1)) - 1)

    T = if bps == 1
        Int8
    elseif bps == 2
        Int16
    elseif bps <= 4
        Int32
    else
        error("Unsupported bit depth: $nbits")
    end

    return T.(clamp.(round.(samples .* maxval), typemin(T), typemax(T)))
end

"""
    write_pcm_samples(io, samples::Matrix, nbits::Integer)

Write PCM sample data in big-endian byte order.
"""
function write_pcm_samples(io::IO, samples::Matrix{T}, nbits::Integer) where {T<:Integer}
    bps = bytes_per_sample(nbits)
    nframes, nch = size(samples)

    for i in 1:nframes
        for ch in 1:nch
            if bps == 1
                write_be(io, Int8(samples[i, ch]))
            elseif bps == 2
                write_be(io, Int16(samples[i, ch]))
            elseif bps == 3
                # Write 3 bytes big-endian
                v = Int32(samples[i, ch])
                write(io, UInt8((v >> 16) & 0xFF))
                write(io, UInt8((v >> 8) & 0xFF))
                write(io, UInt8(v & 0xFF))
            elseif bps == 4
                write_be(io, Int32(samples[i, ch]))
            end
        end
    end
end

# ============================================================================
# Chunk Parsers
# ============================================================================

"""
    read_comm_chunk(io, cksize) -> AIFFFormat

Parse the Common Chunk ('COMM').
"""
function read_comm_chunk(io::IO, cksize::Int)
    nchannels = read_be(io, Int16)
    nframes = read_be(io, UInt32)
    samplebits = read_be(io, Int16)
    samplerate = read_ieee_extended(io)

    # If cksize > 18, skip remaining bytes (AIFF-C extension)
    remaining = cksize - 18
    if remaining > 0
        skip(io, remaining)
    end

    return AIFFFormat(nchannels, nframes, samplebits, samplerate)
end

"""
    read_ssnd_chunk(io, cksize, format) -> Matrix

Parse the Sound Data Chunk ('SSND').
"""
function read_ssnd_chunk(io::IO, cksize::Int, fmt::AIFFFormat)
    offset = read_be(io, UInt32)
    blocksize = read_be(io, UInt32)
    return read_pcm_samples(io, fmt, offset)
end

"""
    read_marker_chunk(io, cksize) -> Vector{AIFFMarker}

Parse the Marker Chunk ('MARK').
"""
function read_marker_chunk(io::IO, cksize::Int)
    start_pos = position(io)
    nummarkers = read_be(io, UInt16)
    markers = AIFFMarker[]

    for _ in 1:nummarkers
        id = read_be(io, Int16)
        pos = read_be(io, UInt32)
        # pstring: 1-byte count followed by text
        namelen = read(io, UInt8)
        namebytes = read(io, namelen)
        name = String(namebytes)
        # Pad byte if the total (count + text) is odd
        if (namelen + 1) % 2 != 0
            skip(io, 1)
        end
        push!(markers, AIFFMarker(id, pos, name))
    end

    # Skip any remaining bytes
    consumed = position(io) - start_pos
    remaining = cksize - consumed
    if remaining > 0
        skip(io, remaining)
    end

    return markers
end

"""
    read_inst_chunk(io, cksize) -> AIFFInstrument

Parse the Instrument Chunk ('INST').
"""
function read_inst_chunk(io::IO, cksize::Int)
    basenote = read_be(io, Int8)
    detune = read_be(io, Int8)
    lownote = read_be(io, Int8)
    highnote = read_be(io, Int8)
    lowvel = read_be(io, Int8)
    highvel = read_be(io, Int8)
    gain = read_be(io, Int16)

    sustain = AIFFLoop(read_be(io, Int16), read_be(io, Int16), read_be(io, Int16))
    release = AIFFLoop(read_be(io, Int16), read_be(io, Int16), read_be(io, Int16))

    return AIFFInstrument(basenote, detune, lownote, highnote, lowvel, highvel, gain, sustain, release)
end

"""
    read_text_chunk(io, cksize) -> String

Parse a text chunk ('NAME', 'AUTH', '(c) ', 'ANNO').
"""
function read_text_chunk(io::IO, cksize::Int)
    bytes = read(io, cksize)
    # Pad byte for odd-sized chunks is handled by the main loop
    return String(bytes)
end

# ============================================================================
# Main Read Function
# ============================================================================

"""
    aiffread(filename; format="double")

Read an AIFF file and return `(samples, samplerate, nbits, extra)`.

# Arguments
- `filename`: Path to the AIFF file.
- `format`: Output format for sample data:
  - `"double"`: `Matrix{Float64}` normalized to [-1.0, 1.0] (default)
  - `"native"`: Raw integer PCM values
  - `"size"`: Return only `(nframes, nchannels)` without reading sample data

# Returns
A named tuple `(data, samplerate, nbits, extra)` where:
- `data`: Sample matrix (rows=frames, cols=channels), or tuple of dimensions
- `samplerate`: Sample rate in Hz (Float64)
- `nbits`: Bit depth
- `extra`: Dict with metadata (:markers, :instrument, :name, :author, :copyright, :annotations, :comments, :chunks)
"""
function aiffread(filename::AbstractString; format::AbstractString="double")
    open(filename, "r") do io
        return aiffread(io; format=format)
    end
end

function aiffread(io::IO; format::AbstractString="double")
    # Read FORM header
    form_id = read_chunkid(io)
    form_id === :FORM || error("Not an IFF file: expected 'FORM', got '$(form_id)'")

    form_size = read_be(io, UInt32)
    form_type = read_chunkid(io)
    form_type === :AIFF || error("Not an AIFF file: expected 'AIFF', got '$(form_type)'")

    # State for chunks
    comm = nothing
    samples = nothing
    extra = Dict{Symbol,Any}()
    extra[:markers] = AIFFMarker[]
    extra[:annotations] = String[]
    extra[:chunks] = AIFFChunk[]

    form_end = position(io) + Int(form_size) - 4  # -4 for formType already read

    while position(io) < form_end
        chunk_start = position(io)
        ckid = read_chunkid(io)
        cksize = Int(read_be(io, UInt32))
        data_start = position(io)

        if ckid === :COMM
            comm = read_comm_chunk(io, cksize)
        elseif ckid === :SSND
            if format == "size"
                skip(io, cksize)
            else
                comm === nothing && error("SSND chunk found before COMM chunk")
                samples = read_ssnd_chunk(io, cksize, comm)
            end
        elseif ckid === :MARK
            extra[:markers] = read_marker_chunk(io, cksize)
        elseif ckid === :INST
            extra[:instrument] = read_inst_chunk(io, cksize)
        elseif ckid === :NAME
            extra[:name] = read_text_chunk(io, cksize)
        elseif ckid === :AUTH
            extra[:author] = read_text_chunk(io, cksize)
        elseif ckid === Symbol("(c) ")
            extra[:copyright] = read_text_chunk(io, cksize)
        elseif ckid === :ANNO
            push!(extra[:annotations], read_text_chunk(io, cksize))
        else
            # Unknown chunk: preserve raw data
            data = read(io, cksize)
            push!(extra[:chunks], AIFFChunk(ckid, data))
        end

        # Seek to end of chunk data (+ pad byte if odd size)
        next_pos = data_start + cksize
        if cksize % 2 != 0
            next_pos += 1
        end
        seek(io, min(next_pos, form_end))
    end

    comm === nothing && error("AIFF file missing required COMM chunk")

    if format == "size"
        return (data=(Int(comm.nframes), Int(comm.nchannels)),
                samplerate=comm.samplerate,
                nbits=Int(comm.samplebits),
                extra=extra)
    end

    if samples === nothing
        # No SSND chunk, but nframes might be 0
        if comm.nframes == 0
            T = format == "native" ? Int16 : Float64
            samples_out = Matrix{T}(undef, 0, Int(comm.nchannels))
        else
            error("AIFF file missing required SSND chunk")
        end
    elseif format == "double"
        samples_out = pcm_to_float(samples, Int(comm.samplebits))
    else
        samples_out = samples
    end

    return (data=samples_out,
            samplerate=comm.samplerate,
            nbits=Int(comm.samplebits),
            extra=extra)
end

# ============================================================================
# Main Write Function
# ============================================================================

"""
    aiffwrite(filename, samples, samplerate; nbits=16)

Write audio data to an AIFF file.

# Arguments
- `filename`: Output file path.
- `samples`: Audio data as a `Matrix` (rows=frames, cols=channels).
  If `Float64`, values should be in [-1.0, 1.0].
  If integer types, values are written directly as PCM.
- `samplerate`: Sample rate in Hz.
- `nbits`: Bit depth (8, 16, 24, or 32). Default is 16.
"""
function aiffwrite(filename::AbstractString, samples::AbstractMatrix, samplerate::Real; nbits::Integer=16)
    open(filename, "w") do io
        aiffwrite(io, samples, samplerate; nbits=nbits)
    end
end

function aiffwrite(io::IO, samples::AbstractMatrix{T}, samplerate::Real; nbits::Integer=16) where {T}
    nframes, nchannels = size(samples)
    bps = bytes_per_sample(nbits)

    # Convert float samples to PCM integers if necessary
    pcm_samples = if T <: AbstractFloat
        float_to_pcm(Float64.(samples), nbits)
    else
        samples
    end

    # Calculate sizes
    ssnd_data_size = nframes * nchannels * bps + 8  # +8 for offset & blockSize
    comm_size = 18
    total_size = 4 + (8 + comm_size) + (8 + ssnd_data_size)  # formType + COMM + SSND
    # Pad SSND if odd
    if ssnd_data_size % 2 != 0
        total_size += 1
    end

    # --- FORM header ---
    write_chunkid(io, :FORM)
    write_be(io, UInt32(total_size))
    write_chunkid(io, :AIFF)

    # --- COMM chunk ---
    write_chunkid(io, :COMM)
    write_be(io, UInt32(comm_size))
    write_be(io, Int16(nchannels))
    write_be(io, UInt32(nframes))
    write_be(io, Int16(nbits))
    write_ieee_extended(io, Float64(samplerate))

    # --- SSND chunk ---
    write_chunkid(io, :SSND)
    write_be(io, UInt32(ssnd_data_size))
    write_be(io, UInt32(0))  # offset
    write_be(io, UInt32(0))  # blockSize
    write_pcm_samples(io, pcm_samples, nbits)

    # Pad byte if odd
    if ssnd_data_size % 2 != 0
        write(io, UInt8(0))
    end

    return nothing
end

# Convenience: accept 1D vector as mono audio
function aiffwrite(filename::AbstractString, samples::AbstractVector, samplerate::Real; nbits::Integer=16)
    aiffwrite(filename, reshape(samples, :, 1), samplerate; nbits=nbits)
end

function aiffwrite(io::IO, samples::AbstractVector, samplerate::Real; nbits::Integer=16)
    aiffwrite(io, reshape(samples, :, 1), samplerate; nbits=nbits)
end

end # module AIFF
