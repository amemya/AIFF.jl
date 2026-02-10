# macOS AudioQueue playback via Core Audio
module AIFFPlay
import ..aiffplay

const OSStatus = Int32
const CFTypeRef = Ptr{Cvoid}
const CFRunLoopRef = Ptr{Cvoid}
const CFStringRef = Ptr{Cvoid}
const AudioQueueRef = Ptr{Cvoid}

# ============================================================================
# Core Audio Constants
# ============================================================================

# Format ID: 'lpcm' (Linear PCM)
const kAudioFormatLinearPCM =
    UInt32('l') << 24 | UInt32('p') << 16 | UInt32('c') << 8 | UInt32('m')

# Format flags
const kAudioFormatFlagIsFloat               = (1 << 0)
const kAudioFormatFlagIsBigEndian           = (1 << 1)
const kAudioFormatFlagIsSignedInteger       = (1 << 2)
const kAudioFormatFlagIsPacked              = (1 << 3)
const kAudioFormatFlagIsAlignedHigh         = (1 << 4)
const kAudioFormatFlagIsNonInterleaved      = (1 << 5)
const kAudioFormatFlagIsNonMixable          = (1 << 6)
const kAudioFormatFlagsAreAllClear          = 0

const kNumberBuffers = 3

const CoreFoundation =
    "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
const AudioToolbox =
    "/System/Library/Frameworks/AudioToolbox.framework/Versions/A/AudioToolbox"

# ============================================================================
# Core Audio Struct Definitions
# ============================================================================

struct AudioStreamPacketDescription
    mStartOffset::Int64
    mVariableFramesInPacket::UInt32
    mDataByteSize::UInt32
end

struct SMPTETime
    mSubframes::Int16
    mSubframeDivisor::Int16
    mCounter::UInt32
    mType::UInt32
    mFlags::UInt32
    mHours::Int16
    mMinutes::Int16
    mSeconds::Int16
    mFrames::Int16

    SMPTETime() = new(0, 0, 0, 0, 0, 0, 0, 0, 0)
end

struct AudioTimeStamp
    mSampleTime::Float64
    mHostTime::UInt64
    mRateScalar::Float64
    mWordClockTime::UInt64
    mSMPTETime::SMPTETime
    mFlags::UInt32
    mReserved::UInt32

    AudioTimeStamp(fs) = new(fs, 0, 0, 0, SMPTETime(), 0, 0)
end

mutable struct AudioQueueBuffer
    mAudioDataBytesCapacity::UInt32
    mAudioData::Ptr{Cvoid}
    mAudioDataByteSize::UInt32
    mUserData::Ptr{Cvoid}
    mPacketDescriptionCapacity::UInt32
    mPacketDescription::Ptr{AudioStreamPacketDescription}
    mPacketDescriptionCount::UInt32
end

const AudioQueueBufferRef = Ptr{AudioQueueBuffer}

struct AudioStreamBasicDescription
    mSampleRate::Float64
    mFormatID::UInt32
    mFormatFlags::UInt32
    mBytesPerPacket::UInt32
    mFramesPerPacket::UInt32
    mBytesPerFrame::UInt32
    mChannelsPerFrame::UInt32
    mBitsPerChannel::UInt32
    mReserved::UInt32

    function AudioStreamBasicDescription(fs, fmtID, fmtFlags, bytesPerPacket,
                                         framesPerPacket, bytesPerFrame,
                                         channelsPerFrame, bitsPerChannel)
        new(fs, fmtID, fmtFlags, bytesPerPacket, framesPerPacket,
            bytesPerFrame, channelsPerFrame, bitsPerChannel, 0)
    end
end

mutable struct AudioQueueData{T,N}
    samples::AbstractArray{T,N}
    aq::AudioQueueRef
    offset::Int
    nSamples::Int
    nBuffersEnqueued::UInt
    runLoop::CFRunLoopRef
    callbackPtr::Ptr{Cvoid}  # prevent GC of @cfunction

    function AudioQueueData(samples::AbstractArray{T,N}) where {T,N}
        new{T,N}(
            samples, convert(AudioQueueRef, 0), 0,
            size(samples, 1), 0, convert(CFRunLoopRef, 0),
            C_NULL)
    end
end

# ============================================================================
# CoreFoundation RunLoop
# ============================================================================

CFRunLoopGetCurrent() =
    ccall((:CFRunLoopGetCurrent, CoreFoundation), CFRunLoopRef, ())

CFRunLoopRun() =
    ccall((:CFRunLoopRun, CoreFoundation), Cvoid, ())

CFRunLoopStop(rl) =
    ccall((:CFRunLoopStop, CoreFoundation), Cvoid, (CFRunLoopRef,), rl)

getCoreFoundationRunLoopDefaultMode() =
    unsafe_load(cglobal((:kCFRunLoopDefaultMode, CoreFoundation), CFStringRef))

# ============================================================================
# AudioQueue API Wrappers
# ============================================================================

function AudioQueueFreeBuffer(aq::AudioQueueRef, buf::AudioQueueBufferRef)
    result = ccall((:AudioQueueFreeBuffer, AudioToolbox),
                   OSStatus,
                   (AudioQueueRef, AudioQueueBufferRef), aq, buf)
    result != 0 && error("AudioQueueFreeBuffer failed with $result")
end

function AudioQueueAllocateBuffer(aq::AudioQueueRef,
                                  bufferByteSize::Integer)::AudioQueueBufferRef
    newBuffer = Ref{AudioQueueBufferRef}(0)
    result = ccall((:AudioQueueAllocateBuffer, AudioToolbox), OSStatus,
                   (AudioQueueRef, UInt32, Ref{AudioQueueBufferRef}),
                   aq, bufferByteSize, newBuffer)
    result != 0 && error("AudioQueueAllocateBuffer failed with $result")
    return newBuffer[]
end

function AudioQueueEnqueueBuffer(aq::AudioQueueRef, bufPtr::AudioQueueBufferRef)
    result = ccall((:AudioQueueEnqueueBuffer, AudioToolbox),
                   OSStatus,
                   (AudioQueueRef, AudioQueueBufferRef, UInt32, Ptr{Cvoid}),
                   aq, bufPtr, 0, C_NULL)
    result != 0 && error("AudioQueueEnqueueBuffer failed with $result")
end

function AudioQueueStart(aq)
    result = ccall((:AudioQueueStart, AudioToolbox), OSStatus,
                   (AudioQueueRef, Ptr{AudioTimeStamp}), aq, C_NULL)
    result != 0 && error("AudioQueueStart failed with $result")
end

function AudioQueueStop(aq, immediate)
    result = ccall((:AudioQueueStop, AudioToolbox), OSStatus,
                   (AudioQueueRef, Bool), aq, immediate)
    result != 0 && error("AudioQueueStop failed with $result")
end

function AudioQueueDispose(aq::AudioQueueRef, immediate::Bool)
    result = ccall((:AudioQueueDispose, AudioToolbox),
                   OSStatus,
                   (AudioQueueRef, Bool), aq, immediate)
    result != 0 && error("AudioQueueDispose failed with $result")
end

# ============================================================================
# Buffer Management & Callback
# (Must be defined BEFORE AudioQueueNewOutput, which uses @cfunction)
# ============================================================================

@inline function enqueueBuffer(userData::AudioQueueData{T,N},
                               buf::AudioQueueBufferRef) where {T,N}
    if userData.offset >= userData.nSamples
        return false
    end

    buffer::AudioQueueBuffer = unsafe_load(buf)

    nChannels = size(userData.samples, 2)
    nFrames = buffer.mAudioDataBytesCapacity ÷ (sizeof(T) * nChannels)

    offset = userData.offset
    nFrames = min(nFrames, userData.nSamples - offset)

    coreAudioData = convert(Ptr{T}, buffer.mAudioData)
    if nChannels == 1
        for i in 1:nFrames
            unsafe_store!(coreAudioData, userData.samples[i + offset], i)
        end
    else
        idx = 0
        for i in 1:nFrames
            for j in 1:nChannels
                idx += 1
                unsafe_store!(coreAudioData, userData.samples[i + offset, j], idx)
            end
        end
    end
    buffer.mAudioDataByteSize = nFrames * nChannels * sizeof(T)

    unsafe_store!(buf, buffer)

    userData.offset = offset + nFrames
    userData.nBuffersEnqueued += 1
    AudioQueueEnqueueBuffer(userData.aq, buf)
    return true
end

function playCallback(userData::AudioQueueData{T,N}, aq::AudioQueueRef,
                      buf::AudioQueueBufferRef) where {T,N}
    userData.nBuffersEnqueued -= 1
    if !enqueueBuffer(userData, buf)
        AudioQueueFreeBuffer(aq, buf)
        if userData.nBuffersEnqueued == 0
            AudioQueueStop(aq, false)
            CFRunLoopStop(userData.runLoop)
        end
    end
    return
end

# ============================================================================
# AudioQueueNewOutput (requires playCallback to be defined above)
# ============================================================================

function AudioQueueNewOutput(format::AudioStreamBasicDescription,
                             userData::AudioQueueData{T,N}) where {T,N}
    runLoop = CFRunLoopGetCurrent()
    userData.runLoop = runLoop
    runLoopMode = getCoreFoundationRunLoopDefaultMode()

    newAudioQueue = Ref{AudioQueueRef}(0)
    cCallbackProc = @cfunction(playCallback, Cvoid,
                               (Ref{AudioQueueData{T,N}}, AudioQueueRef, AudioQueueBufferRef))
    # Store callback pointer to prevent GC during playback
    userData.callbackPtr = Base.unsafe_convert(Ptr{Cvoid}, cCallbackProc)
    result = ccall((:AudioQueueNewOutput, AudioToolbox), OSStatus,
                   (Ptr{AudioStreamBasicDescription}, Ptr{Cvoid}, Ref{AudioQueueData{T,N}},
                    CFRunLoopRef, CFStringRef, UInt32, Ref{AudioQueueRef}),
                   Ref(format), cCallbackProc, Ref(userData),
                   runLoop, runLoopMode, 0, newAudioQueue)
    result != 0 && error("AudioQueueNewOutput failed with $result")
    return newAudioQueue[]
end

# ============================================================================
# Format & Play
# ============================================================================

function getFormatFlags(el)
    flags = kAudioFormatFlagsAreAllClear | kAudioFormatFlagIsPacked
    if el <: AbstractFloat
        flags |= kAudioFormatFlagIsFloat
    elseif el <: Integer
        flags |= kAudioFormatFlagIsSignedInteger
    else
        error("Element type $(el) not supported for aiffplay")
    end
    return flags
end

function getFormatForData(data, fs)
    elType = eltype(data)
    fmtFlags = getFormatFlags(elType)
    elSize = sizeof(elType)
    nChannels = size(data, 2)
    return AudioStreamBasicDescription(
        fs, kAudioFormatLinearPCM, fmtFlags,
        elSize * nChannels,  # bytes per packet
        1,                   # frames per packet
        elSize * nChannels,  # bytes per frame
        nChannels,           # channels per frame
        elSize * 8)          # bits per channel
end

function aiffplay(data::AbstractVecOrMat{<:Real}, fs::Real)
    # Ensure 2D and materialize to Array (needed for unsafe_store! pointer access)
    samples = ndims(data) == 1 ? reshape(collect(data), :, 1) : collect(data)

    # Normalize integers to Float32 [-1.0, 1.0]; convert Float64 to Float32
    if eltype(samples) <: Integer
        maxval = Float32(typemax(eltype(samples)))
        samples = Float32.(samples) ./ maxval
    elseif eltype(samples) != Float32
        samples = Float32.(samples)
    end

    userData = AudioQueueData(samples)
    userData.aq = AudioQueueNewOutput(getFormatForData(samples, fs), userData)

    buffers = [AudioQueueAllocateBuffer(userData.aq, 16384) for _ in 1:kNumberBuffers]
    for buf in buffers
        enqueueBuffer(userData, buf)
    end

    AudioQueueStart(userData.aq)
    CFRunLoopRun()
    AudioQueueDispose(userData.aq, true)
end

end # module AIFFPlay
