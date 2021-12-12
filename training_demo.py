import random
import os
import argparse
import gc
import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils.data_loader import Data
from dnn.neural_net import neuralNet
from config import Config
from utils.tools import denormalization


class Scheduler:
    """
    Custom scheduler for exponential decay
    """
    def __init__(self, decay_rate=0.99, decay_step=1):
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def exponential_scheduler(self, step, learning_rate):
        assert step >= 0, print("[Error] Step should not be negative")
        if step % self.decay_step == 0:
            new_lr = learning_rate * self.decay_rate
            print("Learning rate decayed to : {}".format(new_lr))
            return new_lr
        else:
            return learning_rate


class CustomCheckpoint(ModelCheckpoint):
    def _save_model(self, epoch, logs):
        """
        Overriding to modify checkpoint saving frequency
        """
        if isinstance(self.save_freq, int) and (epoch % self.save_freq == 0):
            self.save_freq = 'epoch'  # To prevent a duplicate call during a same epoch
            self.filepath += '{}.h5'.format(str(epoch))
            super()._save_model(epoch, logs)


def train_jinet(nn_cls, continue_epoch, config):
    """
    Train JI-Net
    """
    model_name = 'jinet'
    log_dir = os.path.join(config.log_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)
    tf_log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dataCls = Data(config.seq_path, is_test=False)
    val_x_batch, val_y_batch = dataCls.get_jinet_batch(config.model[model_name]['epoch_batch_len'], 'validation')
    validate_sample(val_x_batch, val_y_batch, None, 5, model_name, log_dir)

    # Create validation batch
    val_idx = [i for i in range(len(val_x_batch))]
    random.shuffle(val_idx)

    scheduler = Scheduler(decay_rate=0.99, decay_step=config.model[model_name]['repeat'])
    lr_callback = LearningRateScheduler(scheduler.exponential_scheduler)
    sgd = SGD(learning_rate=config.model[model_name]['init_lr'], momentum=0.9)
    nn_cls.nets[model_name].compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir=tf_log_dir)

    for step in range(continue_epoch, config.model[model_name]['tot_epoch'] + 1, config.model[model_name]['repeat']):
        checkpoint_path = os.path.join(config.model_dir, '{}-'.format(config.model[model_name]['save_name']))
        checkpoint_callback = CustomCheckpoint(filepath=checkpoint_path, save_freq=config.model[model_name]['log_intv'])
        cur_lr = nn_cls.nets[model_name].optimizer.lr.numpy()
        print("Train epoch : {}, Learning rate : {}".format(step, cur_lr))
        train_x_batch, train_y_batch = dataCls.get_jinet_batch(config.model[model_name]['epoch_batch_len'], 'train')

        # Create training batch
        train_idx = [i for i in range(len(train_x_batch))]
        random.shuffle(train_idx)

        nn_cls.nets[model_name].fit(train_x_batch[train_idx], train_y_batch[train_idx], batch_size=config.model[model_name]['train_batch_len'],
                         epochs=step + config.model[model_name]['repeat'], initial_epoch=step,
                         validation_data=(val_x_batch[val_idx], val_y_batch[val_idx]),
                                    validation_freq=config.model[model_name]['log_intv'],
                                    callbacks=[lr_callback, tensorboard_callback, checkpoint_callback])
        gc.collect()  # To prevent memory leak in tf2.1


def train_lstm(nn_cls, continue_epoch, config):
    """
    Train LSTM
    """
    model_name = 'lstm'
    log_dir = os.path.join(config.log_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)
    tf_log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dataCls = Data(config.seq_path)
    val_img_batch, val_shp_batch, val_y_batch, val_trk_len = dataCls.get_deeptama_batch(
        nn_cls.max_trk_len, config.model[model_name]['epoch_batch_len'], 'validation')
    validate_sample(val_img_batch, val_y_batch, val_shp_batch, 5, model_name, log_dir)

    # Create validation batch
    val_x_batch, val_idx = nn_cls.create_lstm_input(val_img_batch, val_shp_batch, val_trk_len)

    scheduler = Scheduler(decay_rate=0.99, decay_step=config.model[model_name]['repeat'])
    lr_callback = LearningRateScheduler(scheduler.exponential_scheduler)
    sgd = SGD(learning_rate=config.model[model_name]['init_lr'], momentum=0.9)
    nn_cls.nets[model_name].compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir=tf_log_dir)

    for step in range(continue_epoch, config.model[model_name]['tot_epoch'] + 1, config.model[model_name]['repeat']):
        checkpoint_path = os.path.join(config.model_dir, '{}-'.format(config.model[model_name]['save_name']))
        checkpoint_callback = CustomCheckpoint(filepath=checkpoint_path, save_freq=config.model[model_name]['log_intv'])
        cur_lr = nn_cls.nets[model_name].optimizer.lr.numpy()
        print("Train epoch : {}, Learning rate : {}".format(step, cur_lr))
        train_img_batch, train_shp_batch, train_label_batch, train_trk_len = dataCls.get_deeptama_batch(
            nn_cls.max_trk_len, config.model[model_name]['epoch_batch_len'], 'train')

        # Create training batch
        train_input_batch, train_idx = nn_cls.create_lstm_input(train_img_batch, train_shp_batch, train_trk_len)
        nn_cls.nets[model_name].fit(train_input_batch[train_idx], train_label_batch[train_idx],
                                    batch_size=config.model[model_name]['epoch_batch_len'],
                                    epochs=step + config.model[model_name]['repeat'], initial_epoch=step,
                                    validation_data=(val_x_batch[val_idx], val_y_batch[val_idx]),
                                    validation_freq=config.model[model_name]['log_intv'],
                                callbacks=[lr_callback, tensorboard_callback, checkpoint_callback])
        gc.collect()  # To prevent memory leak in tf2.1


def validate_sample(inputs, labels, shape=None, max_num=5, mode='lstm', save_dir='.'):
    """
    Validate input data samples
    :param inputs: input data array
    :param labels: input label array
    :param shape: shape data for LSTM
    :param max_num: number of samples to validate
    :param mode: 'lstm' or 'jinet'
    :param save_dir: save directory
    :return: None
    """
    visualizer = np.zeros((0, 128 if mode == 'jinet' else 64*(1+inputs.shape[1]), 3))
    pos_neg_cnt = [0, 0]
    colors = [(255, 0, 0), (0, 0, 255)]
    symbols = ['+', '-']

    for i in range(inputs.shape[0]):
        label = np.argmax(labels[i])
        if pos_neg_cnt[label] > max_num:
            continue
        pos_neg_cnt[label] += 1
        templates = np.zeros((128, 0, 3))
        if mode == 'jinet':
            for j in range(2):
                template = denormalization(inputs[i, :, :, j*3:(j+1)*3])
                template = cv2.rectangle(template, (0, 0), (63, 127), colors[label], thickness=2)
                templates = np.concatenate((templates, template), axis=1)
        elif mode == 'lstm':
            template = denormalization(inputs[i, -1, :, :, 3:])
            draw_shp_ratio(template, 0, 0, 5)
            templates = cv2.rectangle(template, (0, 0), (63, 127), colors[label], thickness=2)
            for j in range(inputs.shape[1]):
                template = denormalization(inputs[i, j, :, :, :3])
                draw_shp_ratio(template, shape[i, j, 1], shape[i, j, 2], 5)
                template = cv2.rectangle(template, (0, 0), (63, 127), colors[label], thickness=2)
                templates = np.concatenate((templates, template), axis=1)
        else:
            raise NotImplementedError
        cv2.putText(templates, symbols[label], (5, 20), cv2.FONT_HERSHEY_PLAIN, 2, colors[label], 2, cv2.LINE_AA)
        visualizer = np.concatenate((visualizer, templates), axis=0)

    cv2.imwrite(os.path.join(save_dir, 'debug_data.jpg'), visualizer)


def draw_shp_ratio(template, w_ratio, h_ratio, line=5):
    """
    Draw shape data on image
    :param template: image to validate
    :param w_ratio: relative width ratio
    :param h_ratio: relative height ratio
    :param line: line width
    :return: image
    """
    w_ratio = np.clip(w_ratio+1.0, 0.0, 2.0)
    h_ratio = np.clip(h_ratio+1.0, 0.0, 2.0)
    template = cv2.rectangle(template, (0, 127-line), (63, 127), color=(255, 255, 255), thickness=-1)  # fill white first
    template = cv2.rectangle(template, (63 - line, 0), (63, 127), color=(255, 255, 255), thickness=-1)
    template = cv2.rectangle(template, (0, 127 - line), (int(63 - line), 127), color=(0, 0, 0),
                             thickness=-1)  # fill reference shape bar
    template = cv2.rectangle(template, (63 - line, 0), (63, int(127 - line)), color=(0, 0, 0),
                             thickness=-1)
    template = cv2.rectangle(template, (int(0.5*(63 - line)*min(1, w_ratio)), 127 - line),
                             (int(0.5*(63 - line)*max(1, w_ratio)), 127), color=(0, 0, 255), thickness=-1)
    template = cv2.rectangle(template, (63 - line, int(0.5*(127 - line)*min(1, h_ratio))),
                             (63, int(0.5*(127 - line)*max(1, h_ratio))), color=(0, 0, 255), thickness=-1)
    template = cv2.rectangle(template, (int(0.5 * (63 - line) - 1), 127 - line*2),
                             (int(0.5 * (63 - line) + 1), 127), color=(0, 0, 255), thickness=-1)  # fill shap bar
    template = cv2.rectangle(template, (63 - line*2, int(0.5 * (127 - line) - 1)),
                             (63, int(0.5 * (127 - line) + 1)), color=(0, 0, 255), thickness=-1)

    return template


def draw_graph(step_list, loss_list, acc_list, graph_name, save_dir):
    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].plot(step_list, loss_list, 'tab:green')
    axs[0].set_title('validation loss')
    axs[1].plot(step_list, acc_list, 'tab:orange')
    axs[1].set_title('validation acc')

    fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(save_dir, '{}.png'.format(graph_name)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jinet', '-tj', action='store_true', help='Train jinet')
    parser.add_argument('--train_lstm', '-tl', action='store_true', help='Train lstm')
    parser.add_argument('--jinet_continue_epoch', '-jce', default=0, type=int,
                        help='Epoch from which JINet training starts')
    parser.add_argument('--lstm_continue_epoch', '-lce', default=0, type=int,
                        help='Epoch from which LSTM training starts')
    cmd = ['-tj', '-tl']
    # cmd = ['-tl']
    opts = parser.parse_args(cmd)

    config = Config()
    if opts.train_jinet:
        # Train JI-Net first
        print("Start JINet training from epoch {}".format(opts.jinet_continue_epoch+1))
        nn_cls = neuralNet(train_mode='jinet', continue_epoch=opts.jinet_continue_epoch, config=config)
        train_jinet(nn_cls, opts.jinet_continue_epoch+1, config)
        del nn_cls
    if opts.train_lstm:
        # Train LSTM using pre-trained JI-Net
        print("Start LSTM training from epoch {}".format(opts.jinet_continue_epoch+1))
        nn_cls = neuralNet(train_mode='lstm', continue_epoch=opts.lstm_continue_epoch, config=config)
        train_lstm(nn_cls, opts.lstm_continue_epoch+1, config)
        del nn_cls


if __name__ == "__main__":
    main()
