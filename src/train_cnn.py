import os
import time

import tensorflow as tf

from src.models.common import get_callbacks, enable_gpu_growth, init_wandb
from src.util.processing.data_prep import extract_test_train_examples, \
    sample_count_print, challenge_test_train, \
    extract_annotation_data
from src.util.processing.pipeline import shuffle_all, DataGenerator, FeedAll, \
    get_model
from src.variables.args import set_args
from src.variables.constants import DATA_DIR, AUDIO_DIR

# Load running config
# cfg = load_config()
cfg = set_args()

# Metric logging
init_wandb(cfg)

# GPU memory growth
enable_gpu_growth(cfg['gpu_mem_grow'])

with tf.device('/CPU:0'):
    t1 = time.time()

    split_file = "train_test_demo" if cfg['demo_mode'] \
        else "ICBHI_challenge_train_test"

    test_fn, train_fn = challenge_test_train(
        os.path.join(DATA_DIR, split_file + ".txt"))

    txt_file_names = test_fn + train_fn

    rec_annotations_dict = {s: extract_annotation_data(s, AUDIO_DIR)[1] for s
                            in txt_file_names}

    # Test/Train split + extraction of frames
    sample_dict = extract_test_train_examples(test_fn, train_fn,
                                              rec_annotations_dict, AUDIO_DIR,
                                              cfg['target_sample_rate'],
                                              cfg['sample_length'],
                                              cfg['stretch_percent'],
                                              cfg['cw_stretch_repeats'],
                                              cfg['vtlp_alpha'],
                                              cfg['c_vtlp_repeats'],
                                              cfg['w_vtlp_repeats'],
                                              cfg['cw_vtlp_repeats'],
                                              cfg['vtlp_upper_freq'],
                                              cfg['n_filters'],
                                              cfg['window_length'],
                                              cfg['window_step'],
                                              cfg['window_type'],
                                              cfg['low_freq_cutoff'],
                                              cfg['high_freq_cutoff'],
                                              cfg['sample_tiling'])
    training_clips = sample_dict[0]
    test_clips = sample_dict[1]

    print('[Training set]')
    sample_count_print(training_clips)
    print('[Test set]')
    sample_count_print(test_clips)

    cfg['sample_height'] = training_clips['none'][0][0].shape[0]
    cfg['sample_width'] = training_clips['none'][0][0].shape[1]
    print('Input HxW:{}x{}'.format(cfg['sample_height'], cfg['sample_width']))

    [none_train, c_train, w_train, c_w_train] = [training_clips['none'],
                                                 training_clips['crackles'],
                                                 training_clips['wheezes'],
                                                 training_clips['both']]
    [none_test, c_test, w_test, c_w_test] = [test_clips['none'],
                                             test_clips['crackles'],
                                             test_clips['wheezes'],
                                             test_clips['both']]

    shuffle_all([none_train, c_train, w_train, c_w_train])

    # Data generators
    train_gen = DataGenerator([none_train, c_train, w_train, c_w_train],
                              [1, 1, 1, 1])
    test_gen = FeedAll([none_test, c_test, w_test, c_w_test])

    model = get_model(cfg['model_name'], cfg['sample_height'],
                      cfg['sample_width'], cfg['lr'])

    with tf.device("/GPU:0"):
        stats = model.fit(
            train_gen.generate(cfg['batch_size'], cfg['sample_height'],
                               cfg['sample_width']),
            epochs=cfg['n_epochs'],
            steps_per_epoch=train_gen.n_available_samples() // cfg[
                'batch_size'],
            validation_data=test_gen.generate(cfg['batch_size'],
                                              cfg['sample_height'],
                                              cfg['sample_width']),
            validation_steps=test_gen.n_available_samples() // cfg[
                'batch_size'],
            callbacks=get_callbacks(model, cfg, test_gen.fetch_all_gen(
                cfg['sample_height'], cfg['sample_width'])),
            verbose=2
        )
    t2 = time.time()
    print("RUNNING TIME: {}".format(t2 - t1))
    print("Done.")
