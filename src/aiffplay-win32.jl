# Windows PlaySound playback via Winmm.dll
module AIFFPlay
import ..aiffplay

# ============================================================================
# Win32 Constants
# ============================================================================

const BOOL = Cint
const DWORD = Culong
const TRUE = 1

# PlaySound flags from Winmm.h
const SND_SYNC      = 0x0
const SND_ASYNC     = 0x1
const SND_NODEFAULT = 0x2
const SND_MEMORY    = 0x4

# ============================================================================
# In-memory WAV builder (minimal, for PlaySound SND_MEMORY)
# ============================================================================

"""
Build a minimal in-memory WAV file from Float64 samples.
PlaySound on Windows needs a complete WAV in memory.
"""
function build_wav_memory(data::AbstractMatrix{<:Real}, fs::Real)
    nframes, nchannels = size(data)
    nbits = 16
    bytesPerSample = nbits ÷ 8
    blockAlign = nchannels * bytesPerSample
    byteRate = round(Int, fs) * blockAlign
    dataSize = nframes * blockAlign

    buf = IOBuffer()

    # RIFF header
    write(buf, b"RIFF")
    write(buf, htol(UInt32(36 + dataSize)))
    write(buf, b"WAVE")

    # fmt chunk
    write(buf, b"fmt ")
    write(buf, htol(UInt32(16)))          # chunk size
    write(buf, htol(UInt16(1)))           # PCM format
    write(buf, htol(UInt16(nchannels)))
    write(buf, htol(UInt32(round(UInt32, fs))))
    write(buf, htol(UInt32(byteRate)))
    write(buf, htol(UInt16(blockAlign)))
    write(buf, htol(UInt16(nbits)))

    # data chunk
    write(buf, b"data")
    write(buf, htol(UInt32(dataSize)))

    # Interleaved 16-bit PCM samples (little-endian)
    maxval = Float64(typemax(Int16))
    for i in 1:nframes
        for ch in 1:nchannels
            s = Float64(data[i, ch])
            s = isfinite(s) ? s : 0.0
            s = clamp(s, -1.0, 1.0)
            sample = round(Int16, s * maxval)
            write(buf, htol(sample))
        end
    end

    return take!(buf)
end

# ============================================================================
# Playback
# ============================================================================

function aiffplay(data::AbstractVecOrMat{<:Real}, fs::Real)
    samples = ndims(data) == 1 ? reshape(data, :, 1) : data
    wav = build_wav_memory(Float64.(samples), fs)

    success = ccall((:PlaySoundA, "Winmm.dll"), stdcall, BOOL,
                    (Ptr{Cvoid}, Ptr{Cvoid}, DWORD),
                    wav, C_NULL, SND_MEMORY | SND_SYNC | SND_NODEFAULT)
    Base.windowserror("PlaySound", success == 0)
end

end # module AIFFPlay
