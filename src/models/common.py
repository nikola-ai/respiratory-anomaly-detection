import os

import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, log_loss, \
    confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import Callback, EarlyStopping

from src.util.processing.data_prep import challenge_score
from src.variables.config import save_config
from src.variables.constants import MODEL_CHECKPOINT_DIR, CONFIG_CHECKPOINT_DIR


def wb_log_history(history):
    log_dict = {}
    for metric_name in history.keys():
        log_dict[metric_name] = history[metric_name][-1]
    wandb.log(log_dict)


def init_wandb(config):
    if config['demo_mode'] is False:
        wandb.init(config=config, group=config['model_name'], reinit=True)
        wandb.run.save()


def enable_gpu_growth(enabled_growth):
    if enabled_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
            except RuntimeError as e:
                print(e)


def form_save_name(save_dir, model_type, model_name, sensitivity, specificity,
                   score, epoch):
    return save_dir + os.sep + '{}_sc{:.3}_se{:.2}_sp{:.2}_e{}_{}'. \
        format(model_type,
               float(score),
               float(sensitivity),
               float(specificity),
               epoch,
               model_name)


def scores_thresh_pass(model, se_thresh, sp_thresh, sc_thresh):
    return model.se[-1] >= se_thresh and model.sp[-1] >= sp_thresh and \
           model.sc[-1] >= sc_thresh


class SaveModelCallback(Callback):
    def __init__(self, model, config, monitor='sc',
                 model_save_dir=MODEL_CHECKPOINT_DIR,
                 config_save_dir=CONFIG_CHECKPOINT_DIR, verbose=1, mode='max'):
        super().__init__()
        self.model = model
        self.config = config
        self.monitor = monitor
        self.prev_monitor = getattr(self.model, self.monitor)
        self.model_type = config['model_name']
        self.model_save_dir = model_save_dir
        self.config_save_dir = config_save_dir
        self.verbose = verbose
        self.mode = mode
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if scores_thresh_pass(self.model, self.config['se_thresh'],
                              self.config['sp_thresh'],
                              self.config['sc_thresh']):
            # Save model
            self.model.save(
                form_save_name(self.model_save_dir, self.model_type,
                               wandb.run.name, self.model.se[-1],
                               self.model.sp[-1], self.model.sc[-1],
                               self.epoch) + '.h5')
            # Dump config
            save_config(self.config,
                        form_save_name(self.config_save_dir, self.model_type,
                                       wandb.run.name, self.model.se[-1],
                                       self.model.sp[-1], self.model.sc[-1],
                                       self.epoch) + '.yml')
            if self.verbose == 1:
                print('MODEL SAVED!')


class ScoreCallback(Callback):
    def __init__(self, model, config, test_gen, space=None):
        super().__init__()
        self.model = model
        self.config = config
        self.space = space
        self.epoch = 1
        self.test_gen = test_gen
        self.model.sp = []
        self.model.se = []
        self.model.sc = []

    def on_epoch_end(self, epoch, logs=None):
        X_data, y_true = self.test_gen.__next__()
        y_prob = self.model.predict(X_data)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_true, axis=1)
        report = classification_report(y_true, y_pred,
                                       target_names=['none', 'crackles',
                                                     'wheezes', 'both'],
                                       output_dict=True,
                                       zero_division=0)
        micro_avg = report['weighted avg']
        l_loss = log_loss(y_true, y_prob)
        se, sp, sc = challenge_score(y_pred, y_true)
        self.model.se.append(se)
        self.model.sp.append(sp)
        self.model.sc.append(sc)
        print(confusion_matrix(y_true, y_pred))
        print("Log loss: ", l_loss)
        print("Micro avg: ", micro_avg)
        if not self.config['demo_mode']:
            # Current search space
            wandb.log(self.space)
            # Score
            wandb.log({'se': self.model.se[-1], 'sp': self.model.sp[-1],
                       'score:': self.model.sc[-1]})
            # ROC-AUC
            wandb.log({'roc_auc': roc_auc_score(y_true, y_prob,
                                                average='weighted',
                                                multi_class='ovr')})
            # Log loss
            wandb.log({'log_loss': l_loss})
            # Micro F1 (extract from classification report)
            wandb.log(
                {'micro_f1': micro_avg['f1-score'],
                 'precision': micro_avg['precision'],
                 'recall': micro_avg['recall']})
            wandb.log({'epoch': self.epoch})
            # Log models only when scores are above thresholds
            if scores_thresh_pass(self.model, self.config['se_thresh'],
                                  self.config['sp_thresh'],
                                  self.config['sc_thresh']):
                wandb.run.summary['Sensitivity'] = self.model.se[-1]
                wandb.run.summary['Specificity'] = self.model.sp[-1]
                wandb.run.summary['Score'] = self.model.sc[-1]
                # Log model history (losses)
                wb_log_history(self.model.history.history)
        self.epoch += 1


def get_callbacks(model, config, test_gen, space=None):
    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1,
                                   patience=5)
    score_callback = ScoreCallback(model, config, test_gen, space=space)
    save_callback = SaveModelCallback(model, config)
    callbacks = [early_stopping, score_callback]
    if not config['demo_mode']:
        callbacks.append(save_callback)
    return callbacks
