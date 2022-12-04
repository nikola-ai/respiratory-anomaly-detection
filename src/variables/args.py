import argparse

from src.variables.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()

    # Audio
    parser.add_argument('-sr', '--sample_rate',
                        required=False,
                        default=cfg['sample_rate'],
                        type=int,
                        help='Sampling rate')

    parser.add_argument('-tsr', '--target_sample_rate',
                        required=False,
                        default=cfg['sample_rate'],
                        type=int,
                        help='Target sample rate')

    parser.add_argument('-sl', '--sample_length',
                        required=False,
                        default=cfg['sample_length'],
                        type=float,
                        help='Sample length (in seconds)')

    # Spectogram
    parser.add_argument('-wt', '--window_type',
                        required=False,
                        default=cfg['window_type'],
                        help='Window type')

    parser.add_argument('-wl', '--window_length',
                        required=False,
                        type=float,
                        default=cfg['window_length'],
                        help='Window length')

    parser.add_argument('-ws', '--window_step',
                        required=False,
                        type=float,
                        default=cfg['window_step'],
                        help='Sliding window step')

    parser.add_argument('-nf', '--n_filters',
                        required=False,
                        type=int,
                        default=cfg['n_filters'],
                        help='Number of MFCC filters')

    parser.add_argument('-lfc', '--low_freq_cutoff',
                        required=False,
                        type=int,
                        default=cfg['low_freq_cutoff'],
                        help='Low freq cutoff threshold')

    parser.add_argument('-hfc', '--high_freq_cutoff',
                        required=False,
                        type=int,
                        default=cfg['high_freq_cutoff'],
                        help='High freq cutoff threshold')

    parser.add_argument('-sw', '--sample_width',
                        required=False,
                        type=int,
                        default=cfg['sample_width'],
                        help='Audio sample width')

    parser.add_argument('-sh', '--sample_height',
                        required=False,
                        type=int,
                        default=cfg['sample_height'],
                        help='Audio sample height')

    # Augmentation
    parser.add_argument('-vtlp_mf', '--vtlp_upper_freq_min',
                        required=False,
                        type=int,
                        default=cfg['vtlp_upper_freq'][0],
                        help='Vocal tract length perturbation minimum upper '
                             'frequency')

    parser.add_argument('-vtlp_hf', '--vtlp_upper_freq_max',
                        required=False,
                        type=int,
                        default=cfg['vtlp_upper_freq'][1],
                        help='Vocal tract length perturbation maximum upper '
                             'frequency')

    parser.add_argument('-t', '--sample_tiling',
                        required=False,
                        type=bool,
                        default=False,
                        help='Sample tiling')

    # Training
    parser.add_argument('-mg', '--gpu_mem_grow',
                        required=False,
                        type=bool,
                        default=cfg['gpu_mem_grow'],
                        help='GPU dynamic memory growth')

    parser.add_argument('-se', '--se_thresh',
                        required=False,
                        type=float,
                        default=cfg['se_thresh'],
                        help='Sensitivity lowest threshold')

    parser.add_argument('-sp', '--sp_thresh',
                        required=False,
                        type=float,
                        default=cfg['sp_thresh'],
                        help='Specificity lowest threshold')

    parser.add_argument('-m', '--model_name',
                        required=False,
                        default=cfg['model_name'],
                        help='Model name')

    parser.add_argument('-d', '--demo_mode',
                        required=False,
                        type=bool,
                        default=cfg['demo_mode'],
                        help='Demo mode (can omit logging and reduce dataset)')

    parser.add_argument('-lr', '--learning_rate',
                        required=False,
                        type=float,
                        default=cfg['lr'],
                        help='Learning rate')

    parser.add_argument('-b', '--batch_size',
                        required=False,
                        type=int,
                        default=cfg['batch_size'],
                        help='Batch size')

    parser.add_argument('-e', '--n_epochs',
                        required=False,
                        type=int,
                        default=cfg['n_epochs'],
                        help='Number of epochs')

    return parser.parse_args()


def set_args():
    args = parse_args()
    # Audio
    cfg['sample_rate'] = args.sample_rate
    cfg['target_sample_rate'] = args.target_sample_rate
    cfg['sample_length'] = args.sample_length
    # Spectogram
    cfg['window_type'] = args.window_type
    cfg['window_length'] = args.window_length
    cfg['window_step'] = args.window_step
    cfg['n_filters'] = args.n_filters
    cfg['low_freq_cutoff'] = args.low_freq_cutoff
    cfg['high_freq_cutoff'] = args.high_freq_cutoff
    cfg['sample_tiling'] = args.sample_tiling
    # Augmentation
    cfg['vtlp_upper_freq'] = [args.vtlp_upper_freq_min,
                              args.vtlp_upper_freq_max]
    # Training
    cfg['gpu_mem_grow'] = args.gpu_mem_grow
    cfg['se_thresh'] = args.se_thresh
    cfg['sp_thresh'] = args.sp_thresh
    cfg['model_name'] = args.model_name
    cfg['demo_mode'] = args.demo_mode
    cfg['lr'] = args.learning_rate
    cfg['batch_size'] = args.batch_size
    cfg['n_epochs'] = args.n_epochs

    return cfg
