import yaml

from src.variables.constants import DEFAULT_RUNNING_CONFIG

cfg = dict(
    # Audio
    sample_rate=4000,
    target_sample_rate=4000,
    sample_length=5.0,

    # Spectrogram
    window_type='hann',
    window_length=25,  # ms
    window_step=10,  # ms
    n_filters=50,  # mel banks
    low_freq_cutoff=50,  # Hz
    high_freq_cutoff=2000,  # Hz
    sample_width=None,
    sample_height=None,

    # Augmentation
    stretch_percent=20,
    cw_stretch_repeats=2,
    c_vtlp_repeats=3,
    w_vtlp_repeats=4,
    cw_vtlp_repeats=7,

    roll_fft=False,
    vtlp_alpha=[0.9, 1.1],
    vtlp_upper_freq=[3200, 3800],

    # Runtime config
    demo_mode=True,
    gpu_mem_grow=True,
    se_thresh=0.4,
    sp_thresh=0.4,
    sc_thresh=0.5,
    model_name='speech',
    lr=0.0003,
    batch_size=128,
    n_epochs=35,
)


def load_config(path=DEFAULT_RUNNING_CONFIG):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def save_config(config, save_path):
    with open(save_path, 'w') as outfile:
        yaml.dump(config, outfile)
