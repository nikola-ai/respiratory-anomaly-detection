import numpy as np

from src.models.kernel_model import get_kernel_model
from src.models.speech_model import get_speech_model
from src.visualization.cnn_io import Logger


class DataGenerator:
    # sound_clips = [[none],[crackles],[wheezes],[both]]
    # strides: how far the sampling index for each category has advanced
    def __init__(self, sound_clips, strides, roll=False, log_inputs=False):
        self.clips = sound_clips
        self.strides = strides
        self.lengths = [len(arr) for arr in sound_clips]
        self.cursor = [0, 0, 0, 0]
        self.roll = roll
        self.log_inputs = log_inputs
        self.logger = Logger() if log_inputs else None

    def n_available_samples(self):
        return int(min(np.divide(self.lengths, self.strides))) * 4

    # Transpose and wrap each array along the time axis
    def roll_FFT(self, fft_info):
        fft = fft_info[0]
        n_col = fft.shape[1]
        pivot = np.random.randint(n_col)
        return (np.roll(fft, pivot, axis=1)), fft_info[1]

    def generate(self, batch_size, sample_height, sample_width):
        cursor = [0, 0, 0, 0]
        while True:
            i = 0
            X, y = [], []
            for c in range(batch_size):
                cat_length = self.lengths[i]
                cat_clips = self.clips[i]
                cat_stride = self.strides[i]
                cat_advance = np.random.randint(low=1, high=cat_stride + 1)
                clip = cat_clips[(cursor[i] + cat_advance) % cat_length]
                cursor[i] = \
                    (cursor[i] + self.strides[i]) % cat_length  # update cursor
                if self.roll:
                    s = (self.roll_FFT(clip))
                else:
                    s = clip
                X.append(s[0])
                y.append(s[1])
                i = (i + 1) % 4  # go to next class
                if self.log_inputs:
                    self.logger.log_input(X[-1], y[-1], 'train_in')
            X = np.reshape(X, (batch_size, sample_height, sample_width, 1))
            y = np.reshape(y, (batch_size, 4))
            # Shuffle examples
            shuffled_indices = np.arange(batch_size)
            np.random.shuffle(shuffled_indices)
            yield X[shuffled_indices], y[shuffled_indices]


# Used for validation set
class FeedAll:
    def __init__(self, sound_clips, log_inputs=False):
        self.merged = []
        for arr in sound_clips:
            self.merged.extend(arr)
        np.random.shuffle(self.merged)
        self.clips = self.merged
        self.n_clips = len(self.merged)
        self.log_inputs = log_inputs
        self.logger = Logger() if log_inputs else None

    def n_available_samples(self):
        return len(self.clips)

    def fetch_all_gen(self, sample_height, sample_width):
        X_c, y_c = zip(*self.clips)
        while True:
            yield (
                np.reshape(X_c,
                           (self.n_clips, sample_height, sample_width, 1)),
                np.reshape(y_c, (self.n_clips, 4)))

    def generate(self, batch_size, sample_height, sample_width):
        i = 0
        while True:
            X, y = [], []
            for b in range(batch_size):
                clip = self.clips[i]
                i = (i + 1) % self.n_clips
                X.append(clip[0])
                y.append(clip[1])
                if self.log_inputs:
                    self.logger.log_input(X[-1], y[-1], 'test_in')

            yield (np.reshape(X, (batch_size, sample_height, sample_width, 1)),
                   np.reshape(y, (batch_size, 4)))


def sort_cw_cycles(rec_annotations_dict):
    no_label_list, crack_list, wheeze_list, both_sym_list, filename_list = \
        [], [], [], [], []

    for f in rec_annotations_dict.keys():
        d = rec_annotations_dict[f]
        no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
        n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
        n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
        both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
        no_label_list.append(no_labels)
        crack_list.append(n_crackles)
        wheeze_list.append(n_wheezes)
        both_sym_list.append(both_sym)
        filename_list.append(f)

    return no_label_list, crack_list, wheeze_list, both_sym_list, filename_list


def get_model(model_name, input_height, input_width, learning_rate):
    if model_name == 'speech':
        return get_speech_model(input_height, input_width, learning_rate)
    elif model_name == 'kernel':
        return get_kernel_model(input_height, input_width, learning_rate)
    else:
        raise Exception('Model not selected!')


def shuffle_all(shuffle_list):
    for x in shuffle_list:
        np.random.shuffle(x)
