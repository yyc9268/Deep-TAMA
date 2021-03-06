from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation, Softmax
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import data_loader as dl
import os
import random


class neuralNet:
    def __init__(self, is_test=False, max_trk_len=15, train_mode='None'):
        self.max_sequence_len = 0
        self.max_trk_len = max_trk_len
        self.matching_feature_sz = 150

        self._save_dir = 'model'
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

        self._log_dir = 'log'
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

        self.img_input = Input(shape=(128, 64, 6))
        self.lstm_input = Input(shape=(self.max_trk_len, self.matching_feature_sz+3))

        if is_test:
            self.DeepTAMA = load_model(self._save_dir + '/DeepTAMA-model-{}.h5'.format(1000))
            self.JINet = load_model(self._save_dir + '/JINet-model-{}.h5'.format(1000))
            self.featureExtractor = Model(inputs=self.JINet.inputs, outputs=self.JINet.get_layer('matching_feature').output)
        else:
            if train_mode == 'JINet':
                self.JINet = Model(inputs=self.img_input, outputs=self.jinet_output())
            elif train_mode == 'LSTM':
                self.JINet = load_model(self._save_dir + '/JINet-model-{}.h5'.format(1000))
                self.featureExtractor = Model(inputs=self.JINet.inputs,
                                              outputs=self.JINet.get_layer('matching_feature').output)
                self.DeepTAMA = Model(inputs=self.lstm_input, outputs=self.deeptama_output())
            else:
                raise RuntimeError('No such mode exists')

    def __del__(self):
        print("neural-net deleted")

    def jinet_output(self):
        encode1 = Conv2D(filters=12, kernel_size=9)(self.img_input)
        bnorm1 = BatchNormalization()(encode1)
        relu1 = Activation('relu')(bnorm1)
        pool1 = MaxPool2D()(relu1)
        encode2 = Conv2D(filters=16, kernel_size=5)(pool1)
        bnorm2 = BatchNormalization()(encode2)
        relu2 = Activation('relu')(bnorm2)
        pool2 = MaxPool2D()(relu2)
        encode3 = Conv2D(filters=24, kernel_size=5)(pool2)
        bnorm3 = BatchNormalization()(encode3)
        relu3 = Activation('relu')(bnorm3)
        pool3 = MaxPool2D()(relu3)
        flattened = Flatten()(pool3)
        encode4 = Dense(1152, activation='relu')(flattened)
        encode5 = Dense(self.matching_feature_sz, activation='relu', name='matching_feature')(encode4)
        raw_likelihood = Dense(2, activation='relu')(encode5)
        likelihood = Softmax()(raw_likelihood)

        return likelihood

    def deeptama_output(self):
        encode1 = Dense(152, activation='tanh')(self.lstm_input)
        lstm_out = RNN(LSTMCell(128), return_sequences=False, go_backwards=True)(encode1)
        decode1 = Dense(64, activation='tanh')(lstm_out)
        raw_likelihood = Dense(2, activation='relu')(decode1)
        likelihood = Softmax()(raw_likelihood)

        return likelihood

    def train_jinet(self, train_batch_len=128, val_batch_len=128, total_epoch=1020):
        dataCls = dl.data(is_test=False)

        val_x_batch, val_y_batch = dataCls.get_jinet_batch(2048, 'validation')

        # Create validation batch
        val_idx = [i for i in range(len(val_x_batch))]
        random.shuffle(val_idx)

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():
            sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-2)
            self.JINet.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
            step_intv = 4
            loss_list = []
            acc_list = []

            for step in range(0, total_epoch+1, step_intv):
                print("Train step : {}".format(step))
                train_x_batch, train_y_batch = dataCls.get_jinet_batch(2048, 'train')

                # Create training batch
                train_idx = [i for i in range(len(train_x_batch))]
                random.shuffle(train_idx)

                self.JINet.fit(train_x_batch[train_idx], train_y_batch[train_idx], batch_size=train_batch_len, epochs=step+step_intv, initial_epoch=step)

                if step % 20 == 0:
                    val_loss, acc = self.JINet.evaluate(val_x_batch[val_idx], val_y_batch[val_idx], batch_size=val_batch_len)
                    loss_list.append(val_loss)
                    acc_list.append(acc)
                    self.draw_graph([i*20 for i in range(len(loss_list))], loss_list, acc_list, 'jinet')

                    print('{}-step, val_loss : {}, acc : {}'.format(step, val_loss, acc))
                    tf.summary.scalar('siamese validation loss', val_loss, step=step)
                    self.JINet.save(self._save_dir + '/JINet-model-{}.h5'.format(step))

    def train_lstm(self, train_batch_len=128, val_batch_len=128, total_epoch=1000):
        dataCls = dl.data()

        val_img_batch, val_shp_batch, val_label_batch, val_trk_len = dataCls.get_deeptama_batch(self.max_trk_len, 1024, 'validation')
        print(val_img_batch.shape)
        # Create validation batch
        val_input_batch, val_idx = self.create_lstm_input(val_img_batch, val_shp_batch, val_trk_len)

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():
            sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-2)
            self.DeepTAMA.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

            step_intv = 2
            loss_list = []
            acc_list = []

            for step in range(0, total_epoch+1, step_intv):
                print("Training step : {}".format(step))
                train_img_batch, train_shp_batch, train_label_batch, train_trk_len = dataCls.get_deeptama_batch(self.max_trk_len, 256, 'train')

                # Create training batch
                train_input_batch, train_idx = self.create_lstm_input(train_img_batch, train_shp_batch, train_trk_len)

                self.DeepTAMA.fit(train_input_batch[train_idx], train_label_batch[train_idx],
                                  batch_size=train_batch_len, epochs=step+step_intv, initial_epoch=step)

                if step % 20== 0:
                    val_loss, acc = self.DeepTAMA.evaluate(val_input_batch[val_idx], val_label_batch[val_idx], batch_size=val_batch_len)
                    loss_list.append(val_loss)
                    acc_list.append(acc)
                    self.draw_graph([i * 20 for i in range(len(loss_list))], loss_list, acc_list, 'lstm')

                    tf.summary.scalar('validation loss', val_loss, step=step)
                    self.DeepTAMA.save(self._save_dir + '/DeepTAMA-model-{}.h5'.format(step))

    def create_lstm_input(self, img_batch, shp_batch, trk_len):

        # Get pos track features
        JINet_input_batch = np.zeros((sum(trk_len), 128, 64, 6))
        cur_idx = 0
        for i in range(len(img_batch)):
            for j in range(self.max_trk_len - trk_len[i], self.max_trk_len):
                JINet_input_batch[cur_idx, :, :, :] = img_batch[i, j]
                cur_idx += 1

        feature_batch = self.featureExtractor.predict(JINet_input_batch)

        # Create LSTM input batch
        input_batch = np.zeros((len(img_batch), self.max_trk_len, self.matching_feature_sz + 3))
        cur_idx = 0
        for i in range(len(img_batch)):
            j = trk_len[i]
            track_features = feature_batch[cur_idx:cur_idx + j]
            input_batch[i, self.max_trk_len - j:, :-3] = track_features
            input_batch[i, self.max_trk_len - j:, -3:] = shp_batch[i, self.max_trk_len - j:, :]
            cur_idx += j

        shuffled_idx = [i for i in range(len(img_batch))]
        random.shuffle(shuffled_idx)

        return input_batch, shuffled_idx

    def draw_graph(self, step_list, loss_list, acc_list, graph_name):
        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].plot(step_list, loss_list, 'tab:green')
        axs[0].set_title('validation loss')
        axs[1].plot(step_list, acc_list, 'tab:orange')
        axs[1].set_title('validation acc')

        fig.tight_layout(pad=5.0)
        plt.savefig(os.path.join(self._save_dir, '{}.png'.format(graph_name)))

    def get_jinet_likelihood(self, input_pair):
        likelihood = self.featureExtractor.predict(input_pair)

        return likelihood

    def get_feature(self, input_pair):
        feature = self.featureExtractor(input_pair)

        return feature

    def get_likelihood(self, lstm_input):
        likelihood = self.DeepTAMA.predict(lstm_input)

        return likelihood