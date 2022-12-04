import math

import numpy as np

from src.util.processing.sound_io import resample
from src.util.spectrogram.mel_spec import sample2_mel_spectrum


# Creates a copy of each time slice, but stretches or contracts it by a
# random percent < max percentage set
def gen_time_stretch(original, sample_rate, max_percent_change):
    stretch_amount = 1 + np.random.uniform(-1, 1) * (max_percent_change / 100)
    (_, stretched) = resample(sample_rate, original,
                              int(sample_rate * stretch_amount))
    return stretched


# Same as above, but applies it to a list of samples
def augment_list(audio_with_labels, sample_rate, percent_change, n_repeats):
    augmented_samples = []
    for i in range(n_repeats):
        addition = [
            (gen_time_stretch(t[0], sample_rate, percent_change), t[1], t[2])
            for t in audio_with_labels]
        augmented_samples.extend(addition)
    return augmented_samples


def generate_padded_samples(source, output_length, sample_tiling):
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if sample_tiling and frac < 0.5:
        # Tile forward sounds to fill empty space
        cursor = 0
        while (cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    return copy


# Fits each respiratory cycle into a fixed length audio clip, splits may be
# performed and zero padding if necessary
# original:(arr,c,w) -> output:[(arr,c,w),(arr,c,w)]
def split_and_pad(original, desired_length, sample_rate, sample_tiling):
    output_buffer_length = int(desired_length * sample_rate)
    soundclip = original[0]
    n_samples = len(soundclip)
    total_length = n_samples / sample_rate  # length of cycle in seconds
    n_slices = int(math.ceil(
        total_length / desired_length))  # minimum number of slices needed
    samples_per_slice = n_samples // n_slices
    src_start = 0  # staring index to copy from the original buffer
    output = []  # holds the slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples(soundclip[src_start:src_end],
                                       output_buffer_length, sample_tiling)
        output.append((copy, original[1], original[2]))
        src_start += length
    return output


# Takes a list of respiratory cycles, and splits and pads each cycle into
# fixed length buffers (determined by desired_length(sec)), then takes the
# split and padded sample and transforms it into a mel spectrogram
# VTLP_alpha_range = [Lower, Upper] (Bounds of random selection range),
# VTLP_high_freq_range = [Lower, Upper] (-)
# output:[(arr:float[],c:float_bool,w:float_bool),(arr,c,w)]
def split_and_pad_and_apply_mel_spect(original, desired_length, sample_rate,
                                      n_filters, window_length, window_step,
                                      window_type, low_f_cut, high_f_cut,
                                      VTLP_alpha_range=None,
                                      VTLP_high_freq_range=None, n_repeats=1,
                                      sample_tiling=False):
    output = []
    for i in range(n_repeats):
        lst_result = split_and_pad(original, desired_length, sample_rate,
                                   sample_tiling)
        if (VTLP_alpha_range is None) | (VTLP_high_freq_range is None):
            # Do not apply VTLP
            VTLP_params = None
        else:
            # Randomly generate VLTP parameters
            alpha = np.random.uniform(VTLP_alpha_range[0], VTLP_alpha_range[1])
            high_freq = np.random.uniform(VTLP_high_freq_range[0],
                                          VTLP_high_freq_range[1])
            VTLP_params = (alpha, high_freq)
        output.extend([sample2_mel_spectrum(d, sample_rate, n_filters,
                                            VTLP_params, window_length,
                                            window_step,
                                            window_type, low_f_cut, high_f_cut)
                       for d in lst_result])
    return output
