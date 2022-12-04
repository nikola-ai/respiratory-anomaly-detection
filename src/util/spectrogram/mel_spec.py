import math

import numpy as np
import scipy.signal

from src.variables.constants import label2onehot


def freq_to_mel(freq):
    return 1125 * np.log(1 + freq / 700)


def mel_to_freq(mel):
    exponents = mel / 1125
    return 700 * (np.exp(exponents) - 1)


def sample2_mel_spectrum(cycle_info, sample_rate, n_filters, vtlp_params,
                         window_length, window_step, window_type,
                         low_f_cut, high_f_cut):
    f_step = math.ceil(sample_rate / (sample_rate / 1e3 * window_length))
    low_cut = max(0, math.ceil(
        min((sample_rate / 2 / f_step), low_f_cut / f_step)) - 1)
    high_cut = math.ceil(min((sample_rate / 2 / f_step), high_f_cut / f_step))
    nperseg = int(round(window_length * sample_rate / 1e3))
    noverlap = int(round(window_step * sample_rate / 1e3))
    nfft = math.ceil(sample_rate / 1e3 * window_length)
    (f, t, spec) = scipy.signal.spectrogram(cycle_info[0], fs=sample_rate,
                                            nfft=nfft, window=window_type,
                                            nperseg=nperseg, noverlap=noverlap)
    spec = spec[low_cut:high_cut, :].astype(
        np.float32)  # cut out coefficients (frequencies)
    mel_log = \
    FFT2mel_spectrogram(f[low_cut:high_cut], spec, sample_rate, n_filters,
                        vtlp_params)[1]
    mel_min = np.min(mel_log)
    mel_max = np.max(mel_log)
    diff = mel_max - mel_min
    norm_mel_log = (mel_log - mel_min) / diff if (diff > 0) else np.zeros(
        shape=(n_filters, spec.shape[1]))
    if diff == 0:
        print('Error: sample data is completely empty')
    labels = [cycle_info[1], cycle_info[2]]  # crackles, wheezes flags
    return (
        np.reshape(norm_mel_log, (n_filters, spec.shape[1], 1)).astype(
            np.float32),
        # 196x64x1 matrix
        label2onehot(labels))


# vtlp_params = (alpha, f_high)
# Takes an array of the original mel spaced frequencies and returns a
# warped version of them
def VTLP_shift(mel_freq, alpha, f_high, sample_rate):
    nyquist_f = sample_rate / 2
    warp_factor = min(alpha, 1)
    threshold_freq = f_high * warp_factor / alpha
    lower = mel_freq * alpha
    higher = nyquist_f - (nyquist_f - mel_freq) * (
            (nyquist_f - f_high * warp_factor) / (
            nyquist_f - f_high * (warp_factor / alpha)))

    warped_mel = np.where(mel_freq <= threshold_freq, lower, higher)
    return warped_mel.astype(np.float32)


# mel_space_freq: the mel frequencies (HZ) of the filter banks, in addition
# to two maximum and minimum frequency values
# fft_bin_frequencies: the bin frequencies of the FFT output
# Generates a 2d numpy array, with each row containing each filter bank
def generate_mel_filterbanks(mel_space_freq, fft_bin_frequencies):
    n_filters = len(mel_space_freq) - 2
    coeff = []

    # Triangular filter windows
    for mel_index in range(n_filters):
        m = int(mel_index + 1)
        filter_bank = []
        for f in fft_bin_frequencies:
            if f < mel_space_freq[m - 1]:
                hm = 0
            elif f < mel_space_freq[m]:
                hm = (f - mel_space_freq[m - 1]) / (
                        mel_space_freq[m] - mel_space_freq[m - 1])
            elif f < mel_space_freq[m + 1]:
                hm = (mel_space_freq[m + 1] - f) / (
                        mel_space_freq[m + 1] - mel_space_freq[m])
            else:
                hm = 0
            filter_bank.append(hm)
        coeff.append(filter_bank)
    return np.array(coeff, dtype=np.float32)


# Transform spectrogram into mel spectrogram -> (frequencies, spectrum)
# vtlp_params = (alpha, f_high), vtlp will not be applied if set to None
def FFT2mel_spectrogram(f, spec, sample_rate, n_filterbanks, vtlp_params=None):
    (max_mel, min_mel) = (freq_to_mel(max(f)), freq_to_mel(min(f)))
    mel_bins = np.linspace(min_mel, max_mel, num=(n_filterbanks + 2))
    # Convert mel_bins to corresponding frequencies in hz
    mel_freq = mel_to_freq(mel_bins)

    if vtlp_params is None:
        filter_banks = generate_mel_filterbanks(mel_freq, f)
    else:
        # Apply VTLP
        (alpha, f_high) = vtlp_params
        warped_mel = VTLP_shift(mel_freq, alpha, f_high, sample_rate)
        filter_banks = generate_mel_filterbanks(warped_mel, f)

    mel_spectrum = np.matmul(filter_banks, spec)
    return mel_freq[1:-1], np.log10(mel_spectrum + float(10e-12))
