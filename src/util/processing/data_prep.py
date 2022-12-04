import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from src.util.processing.sound_io import read_wav_file, slice_data
from src.util.processing.augmentation import split_and_pad_and_apply_mel_spect, \
    augment_list


# Used to split each individual sound file into separate sound clips
# containing one respiratory cycle each
# output: [filename, (sample_data:np.array, start:float, end:float,
# crackles:bool(float), wheezes:bool(float)) (...) ]
def get_sound_samples(recording_annotations, file_name, root, sample_rate):
    sample_data = [file_name]
    (rate, data) = read_wav_file(os.path.join(root, file_name + '.wav'),
                                 sample_rate)

    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk, start, end, crackles, wheezes))
    return sample_data


def extract_cycles(file_paths, annotation_dict, root, target_rate):
    cycle_list = []
    for file in file_paths:
        data = get_sound_samples(annotation_dict[file], file, root,
                                 target_rate)
        cycles_with_labels = [(d[0], d[3], d[4]) for d in data[1:]]
        cycle_list.extend(cycles_with_labels)
    return cycle_list


def sort_cycles(cycle_list):
    none = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 0))]
    c = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 0))]
    w = [c for c in cycle_list if ((c[1] == 0) & (c[2] == 1))]
    c_w = [c for c in cycle_list if ((c[1] == 1) & (c[2] == 1))]
    return none, c, w, c_w


def augment_block(pool, data, desired_length, target_rate, n_filters,
                  window_length, window_step,
                  window_type, low_f_cut, high_f_cut, VTLP_alpha_range=None,
                  VTLP_high_freq_range=None,
                  n_repeats=1, sample_tiling=False):
    originals = np.concatenate(pool.map(
        partial(split_and_pad_and_apply_mel_spect,
                desired_length=desired_length, sample_rate=target_rate,
                n_filters=n_filters, window_length=window_length,
                window_step=window_step,
                window_type=window_type, low_f_cut=low_f_cut,
                high_f_cut=high_f_cut,
                VTLP_alpha_range=None, VTLP_high_freq_range=None, n_repeats=1,
                sample_tiling=sample_tiling), data))

    augments = np.concatenate(pool.map(
        partial(split_and_pad_and_apply_mel_spect,
                desired_length=desired_length, sample_rate=target_rate,
                n_filters=n_filters, window_length=window_length,
                window_step=window_step,
                window_type=window_type, low_f_cut=low_f_cut,
                high_f_cut=high_f_cut, VTLP_alpha_range=VTLP_alpha_range,
                VTLP_high_freq_range=VTLP_high_freq_range, n_repeats=n_repeats,
                sample_tiling=sample_tiling), data))

    result = np.concatenate((originals, augments))
    # free up memory
    del originals
    del augments

    return result


def extract_test_train_examples(test_file_names, train_file_names,
                                rec_annotations_dict, root, target_rate,
                                desired_length, stretch_percent,
                                cw_stretch_repeats, vtlp_alpha, c_vtlp_repeats,
                                w_vtlp_repeats, cw_vtlp_repeats,
                                vtlp_upper_freq, n_filters, window_length,
                                window_step, window_type, low_f_cut,
                                high_f_cut, sample_tiling):
    # Train classes
    none_train, c_train, w_train, c_w_train = sort_cycles(
        extract_cycles(train_file_names, annotation_dict=rec_annotations_dict,
                       root=root,
                       target_rate=target_rate))

    # Test classes
    none_test, c_test, w_test, c_w_test = sort_cycles(
        extract_cycles(test_file_names, annotation_dict=rec_annotations_dict,
                       root=root,
                       target_rate=target_rate))

    # Training section (Data augmentation procedures)
    # Augment w_only and c_w groups to match the size of c_only
    # no_labels will be artificially reduced in the pipeline  later
    w_stretch = w_train + augment_list(w_train, target_rate, stretch_percent,
                                       cw_stretch_repeats)
    c_w_stretch = c_w_train + augment_list(c_w_train, target_rate,
                                           stretch_percent, cw_stretch_repeats)

    # Speed up and do augmentation tasks in parallel
    # Split up cycles into sound clips with fixed lengths for CNN
    p = Pool(cpu_count())
    # None are not augmented
    train_none = augment_block(p, none_train, desired_length, target_rate,
                               n_filters, window_length, window_step,
                               window_type, low_f_cut, high_f_cut, vtlp_alpha,
                               sample_tiling=sample_tiling)

    # Crackles are not stretched currently
    train_c = augment_block(p, c_train, desired_length, target_rate, n_filters,
                            window_length, window_step,
                            window_type, low_f_cut, high_f_cut, vtlp_alpha,
                            vtlp_upper_freq, c_vtlp_repeats, sample_tiling)

    train_w = augment_block(p, w_stretch, desired_length, target_rate,
                            n_filters, window_length, window_step,
                            window_type, low_f_cut, high_f_cut, vtlp_alpha,
                            vtlp_upper_freq, w_vtlp_repeats, sample_tiling)

    train_c_w = augment_block(p, c_w_stretch, desired_length, target_rate,
                              n_filters, window_length, window_step,
                              window_type, low_f_cut, high_f_cut, vtlp_alpha,
                              vtlp_upper_freq, cw_vtlp_repeats, sample_tiling)

    # Test
    test_none = np.concatenate(
        p.map(partial(split_and_pad_and_apply_mel_spect,
                      desired_length=desired_length, sample_rate=target_rate,
                      n_filters=n_filters, window_length=window_length,
                      window_step=window_step,
                      window_type=window_type, low_f_cut=low_f_cut,
                      high_f_cut=high_f_cut,
                      VTLP_alpha_range=None, VTLP_high_freq_range=None,
                      n_repeats=1, sample_tiling=sample_tiling),
              none_test))
    test_c = np.concatenate(
        p.map(partial(split_and_pad_and_apply_mel_spect,
                      desired_length=desired_length, sample_rate=target_rate,
                      n_filters=n_filters, window_length=window_length,
                      window_step=window_step,
                      window_type=window_type, low_f_cut=low_f_cut,
                      high_f_cut=high_f_cut,
                      VTLP_alpha_range=None, VTLP_high_freq_range=None,
                      n_repeats=1, sample_tiling=sample_tiling),
              c_test))
    test_w = np.concatenate(
        p.map(partial(split_and_pad_and_apply_mel_spect,
                      desired_length=desired_length, sample_rate=target_rate,
                      n_filters=n_filters, window_length=window_length,
                      window_step=window_step,
                      window_type=window_type, low_f_cut=low_f_cut,
                      high_f_cut=high_f_cut,
                      VTLP_alpha_range=None, VTLP_high_freq_range=None,
                      n_repeats=1, sample_tiling=sample_tiling),
              w_test))
    test_c_w = np.concatenate(
        p.map(partial(split_and_pad_and_apply_mel_spect,
                      desired_length=desired_length, sample_rate=target_rate,
                      n_filters=n_filters, window_length=window_length,
                      window_step=window_step,
                      window_type=window_type, low_f_cut=low_f_cut,
                      high_f_cut=high_f_cut,
                      VTLP_alpha_range=None, VTLP_high_freq_range=None,
                      n_repeats=1, sample_tiling=sample_tiling),
              c_w_test))
    p.close()
    p.join()

    train_dict = {'none': train_none, 'crackles': train_c,
                  'wheezes': train_w, 'both': train_c_w}
    test_dict = {'none': test_none, 'crackles': test_c,
                 'wheezes': test_w, 'both': test_c_w}

    return [train_dict, test_dict]


def challenge_test_train(file_path):
    test, train = [], []
    df = pd.read_csv(file_path, sep='\t', header=None)
    for index, row in df.iterrows():
        if row[1] == 'test':
            test.append(row[0])
        else:
            train.append(row[0])
    return test, train


# 0 - None, 1 - Crackle, 2 - Wheeze, 3 - Both
def challenge_score(predictions, labels, verbose=1):
    c = [0, 0, 0, 0]  # corrects
    t = [0, 0, 0, 0]  # totals

    for p, l in zip(predictions, labels):
        t[l] += 1
        if p == l:
            c[p] += 1

    se = (c[1] + c[2] + c[3]) / (t[1] + t[2] + t[3])
    sp = c[0] / t[0]
    sc = (se + sp) / 2

    if verbose == 1:
        print("Se: {:f}, Sp: {:f}\nSCORE: {:f}".format(se, sp, sc))
    elif verbose == 2:
        print("SCORE:", sc)
    return se, sp, sc


def sample_count_print(src_dict):
    print('none:{}\ncrackles:{}\nwheezes:{}\nboth:{}\n'.format(
        len(src_dict['none']),
        len(src_dict['crackles']),
        len(src_dict['wheezes']),
        len(src_dict['both'])))


def extract_annotation_data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data=[tokens],
                                  columns=['Patient number', 'Recording index',
                                           'Chest location',
                                           'Acquisition mode',
                                           'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'),
                                        names=['Start', 'End', 'Crackles',
                                               'Wheezes'], delimiter='\t')
    return recording_info, recording_annotations
