import os

# Data
DATA_DIR = os.path.join('..', 'data', 'raw', 'Respiratory_Sound_Database')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio_and_txt_files/')
EVENTS_DIR = os.path.join(DATA_DIR, 'events')

# Checkpoints
MODEL_CHECKPOINT_DIR = os.path.join('..', 'models')
CONFIG_CHECKPOINT_DIR = os.path.join(MODEL_CHECKPOINT_DIR, 'config_dumps')

# Running config
RUNNING_CONFIG_DIR = os.path.join('.', 'variables')
DEFAULT_RUNNING_CONFIG = os.path.join(RUNNING_CONFIG_DIR, 'config.yml')


# Flattened to onehot labels since the number of combinations is very low
def label2onehot(c_w_flags):
    c = c_w_flags[0]
    w = c_w_flags[1]
    # None
    if (c == False) and (w == False):
        return [1, 0, 0, 0]
    # Crackle
    elif (c == True) and (w == False):
        return [0, 1, 0, 0]
    # Wheeze
    elif (c == False) and (w == True):
        return [0, 0, 1, 0]
    # Both
    elif (c == True) and (w == True):
        return [0, 0, 0, 1]
    else:
        raise Exception("Unknown value to encode!")


def onehot2label(label_vec):
    # None
    if label_vec == [1, 0, 0, 0]:
        return 0, 0
    # Crackle
    elif label_vec == [0, 1, 0, 0]:
        return 1, 0
    # Wheeze
    elif label_vec == [0, 0, 1, 0]:
        return 0, 1
    # Both
    elif label_vec == [0, 0, 0, 1]:
        return 1, 1
    else:
        raise Exception("Unknown value to decode:", label_vec)


def label2str(label):
    if label == (0, 0):
        return 'None'
    elif label == (1, 0):
        return 'Crackle'
    elif label == (0, 1):
        return 'Wheeze'
    elif label == (1, 1):
        return 'Both'
    else:
        raise Exception("Unknown label to decode:", label)
