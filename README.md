# AIFF.jl

A pure Julia package for reading and writing [AIFF (Audio Interchange File Format)](https://en.wikipedia.org/wiki/Audio_Interchange_File_Format) files — no external C libraries required.

## Features

- **Pure Julia** — no dependency on libsndfile or other native libraries
- **Read & Write** uncompressed PCM AIFF files
- **Bit depths**: 8, 16, 24, 32-bit integer PCM
- **Channels**: mono, stereo, and arbitrary multi-channel
- **Metadata**: reads Marker, Instrument, Name, Author, Copyright, and Annotation chunks
- **Round-trip safe**: unknown chunks are preserved as raw bytes

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/amemya/AIFF.jl")
```

## Quick Start

### Reading an AIFF file

```julia
using AIFF

# Returns (data, samplerate, nbits, extra)
result = aiffread("audio.aiff")

samples    = result.data        # Matrix{Float64}, rows=frames, cols=channels
samplerate = result.samplerate  # e.g. 44100.0
nbits      = result.nbits       # e.g. 16
```

The `format` keyword controls the output type:

```julia
# Float64 normalized to [-1.0, 1.0] (default)
result = aiffread("audio.aiff"; format="double")

# Raw integer PCM values (Int8, Int16, or Int32)
result = aiffread("audio.aiff"; format="native")

# Only read dimensions, skip sample data
result = aiffread("audio.aiff"; format="size")
result.data  # (nframes, nchannels)
```

### Writing an AIFF file

```julia
using AIFF

# Generate a 1-second 440 Hz sine wave (mono, 44100 Hz)
t = (0:44099) / 44100
samples = sin.(2π * 440 .* t)

aiffwrite("tone.aiff", samples, 44100; nbits=16)
```

Stereo and multi-channel audio uses a matrix where each column is a channel:

```julia
left  = sin.(2π * 440 .* t)
right = sin.(2π * 880 .* t)
stereo = hcat(left, right)

aiffwrite("stereo.aiff", stereo, 44100; nbits=24)
```

### Accessing metadata

```julia
result = aiffread("audio.aiff")

# Metadata is stored in result.extra (Dict{Symbol, Any})
markers     = result.extra[:markers]      # Vector{AIFFMarker}
annotations = result.extra[:annotations]  # Vector{String}

# Optional fields (present only if the chunk exists in the file)
name       = get(result.extra, :name, nothing)        # String
author     = get(result.extra, :author, nothing)       # String
copyright  = get(result.extra, :copyright, nothing)    # String
instrument = get(result.extra, :instrument, nothing)   # AIFFInstrument
```

## Supported Chunks

| Chunk ID | Type | Description |
|----------|------|-------------|
| `COMM` | Required | Common — channels, sample frames, bit depth, sample rate |
| `SSND` | Required | Sound Data — PCM audio samples |
| `MARK` | Optional | Markers — named positions in the audio |
| `INST` | Optional | Instrument — loop points, MIDI note range, velocity |
| `NAME` | Optional | Name of the sampled sound |
| `AUTH` | Optional | Author / creator |
| `(c) ` | Optional | Copyright notice |
| `ANNO` | Optional | Annotation (free-text comment) |

Unknown chunks are preserved as `AIFFChunk` objects for round-trip fidelity.

## Limitations

- **AIFF only** — AIFF-C (compressed) is not yet supported
- **PCM only** — no floating-point or compressed audio encoding
- **Write** currently outputs only COMM + SSND chunks (no metadata writing yet)

## License

MIT
