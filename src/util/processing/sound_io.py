import wave

import numpy as np
import scipy.io.wavfile as wf


# Will resample all files to the target sample rate and produce a 32bit
# float array
def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode='r')
    (sample_rate, data) = extract_float_arr(wav, str_filename)

    if sample_rate != target_rate:
        (_, data) = resample(sample_rate, data, target_rate)

    wav.close()
    return target_rate, data.astype(np.float32)


def resample(current_rate, data, target_rate):
    x_original = np.linspace(0, 100, len(data))
    x_resampled = np.linspace(0, 100,
                              int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return target_rate, resampled.astype(np.float32)


# Returns (sample_rate, data)
def extract_float_arr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1, 2, 4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1: 255, 2: 32768}
        if bps in [1, 2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))  # clamp to [0.0,1.0]
        return rate, data

    elif bps == 3:
        # 24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception(
            'Unrecognized wave format: {} bytes per sample'.format(bps))


# Note: This function truncates the 24 bit samples to 16 bits of precision
# Reads a wave object returned by the wave.read() method
# Returns the sample rate, as well as the audio in the form of a 32 bit float
# numpy array
# (sample_rate:float, audio_data: float[])
def read24bitwave(lp_wave):
    n_frames = lp_wave.getnframes()
    buf = lp_wave.readframes(n_frames)
    reshaped = np.frombuffer(buf, np.int8).reshape(n_frames, -1)
    short_output = np.empty((n_frames, 2), dtype=np.int8)
    short_output[:, :] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(),
            np.divide(short_output, 32768).reshape(
                -1))  # return numpy array to save memory via array slicing


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())  # bytes per sample
    return bps, lp_wave.getnchannels()


def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]


def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def writewav24(filename, rate, data):
    """Create a 24 bit wav file.
    data must be "array-like", either 1- or 2-dimensional.  If it is 2-d,
    the rows are the frames (i.e. samples) and the columns are the channels.
    The data is assumed to be signed, and the values are assumed to be
    within the range of a 24 bit integer.  Floating point values are
    converted to integers.  The data is not rescaled or normalized before
    writing it to the file.
    Example: Create a 3 second 440 Hz sine wave.
    >>> rate = 22050  # samples per second
    >>> T = 3         # sample duration (seconds)
    >>> f = 440.0     # sound frequency (Hz)
    >>> t = np.linspace(0, T, T*rate, endpoint=False)
    >>> x = (2**23 - 1) * np.sin(2 * np.pi * f * t)
    >>> writewav24("sine24.wav", rate, x)
    """
    a32 = np.asarray(data, dtype=np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
    wavdata = a8.astype(np.uint8).tostring()

    w = wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()
