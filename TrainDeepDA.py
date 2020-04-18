from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Concatenate, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation, Softmax
from tensorflow.keras.backend import sqrt, square, switch, sum, ones_like, print_tensor, constant, maximum, mean, stack, shape, epsilon, squeeze
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import dataLoader as dl
import os
import random


class NeuralNet:

    def __init__(self, is_test=False, train_mode='siamese', max_trk_len=15):
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
        self.lstm_input = Input(shape=(None, 5 * (self.max_trk_len + 1)))

        if is_test:
            self.DeepTAMA = load_model(self._save_dir + '/LSTM-model-{}.h5'.format(1600))
            self.JINet = load_model(self._save_dir + '/JINet-model={}.h5'.format(1000))
            self.featureExtractor = Model(inputs=self.img_input, outputs=self.JINet.get_layer('matching_feature').output)
        else:
            if train_mode == 'JINet':
                self.JINet = Model(inputs=self.img_input, outputs=self.JINet_output)
            elif train_mode == 'LSTM':
                self.JINet = load_model(self._save_dir + '/JINet-model={}.h5'.format(1000))
                self.DeepTAMA = Model(inputs=self.lstm_input, outputs=self.DeepTAMA_output)
            else:
                raise RuntimeError('No such mode exists')

    def JINet_output(self):
        encode1 = Conv2D(filters=12, kernel_size=9, activation='relu')(self.img_input)
        bnorm1 = BatchNormalization()(encode1)
        relu1 = Activation('relu')(bnorm1)
        pool1 = MaxPool2D()(relu1)
        encode2 = Conv2D(filters=16, kernel_size=5, activation='relu')(pool1)
        bnorm2 = BatchNormalization()(encode2)
        relu2 = Activation('relu')(bnorm2)
        pool2 = MaxPool2D()(relu2)
        encode3 = Conv2D(filters=24, kernel_size=5, activation='relu')(pool2)
        pool3 = MaxPool2D()(encode3)
        flattened = Flatten()(pool3)
        encode4 = Dense(1152, activation='relu')(flattened)
        encode5 = Dense(self.matching_feature_sz, activation='relu', name='matching_feature')(encode4)
        raw_likelihood = Dense(2, activation='relu')(encode5)
        likelihood = Softmax()(raw_likelihood)

        return likelihood

    def DeepTAMA_output(self):
        encode1 = Dense(152, activation='tanh')([self.lstm_input])
        lstm_out = RNN(LSTMCell(128), return_sequences=False, go_backwards=True)(encode1)
        decode1 = Dense(64, activation='tanh')(lstm_out)
        raw_likelihood = Dense(2, activation='softmax')(decode1)
        likelihood = Softmax()(raw_likelihood)

        return likelihood

    def trainJINet(self, total_epoch=1000):
        dataCls = dl.data(check_occlusion=True)
        val_batch_len = 128
        train_batch_len = 128
        val_pos, val_neg = dataCls.get_JINet_batch('validation', val_batch_len)

        # Create validation batch
        val_x_batch = np.concatenate((val_pos, val_neg), 0)
        val_y_batch = np.concatenate((np.ones((val_batch_len, 1)), np.ones((val_batch_len, 1))), 0)
        val_idx = [i for i in range(len(val_x_batch))]
        random.shuffle(val_idx)

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():
            self.JINet.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics='accuracy')

            for step in range(1, total_epoch+1):
                print("Train step : {}".format(step))
                train_pos, train_neg = dataCls.get_JINet_batch('train', train_batch_len)

                # Create training batch
                train_x_batch = np.concatenate((train_pos, train_neg), 0)
                train_y_batch = np.concatenate((np.ones((train_batch_len, 1)), np.zeros((train_batch_len, 1))), 0)
                train_idx = [i for i in range(len(train_x_batch))]
                random.shuffle(train_idx)

                self.JINet.fit(train_x_batch[train_idx], train_y_batch[train_idx], batch_size=32, epochs=1)

                if step % 100 == 0:
                    val_loss1, acc1 = self.JINet.evaluate(val_x_batch[val_idx], val_y_batch[val_idx], batch_size=32)
                    print('{}-step, val_loss1 : {}'.format(step, val_loss1, acc1))
                    tf.summary.scalar('siamese validation loss', val_loss1, step=step)
                    self.JINet.save(self._save_dir + '/JINet-model-{}.h5'.format(step))

    def trainLSTM(self, total_epoch=5000):
        dataCls = dl.data()
        val_batch_len = 128
        val_pos_img, val_pos_shp, val_neg_img, val_neg_shp, val_trk_len = dataCls.get_LSTM_batch(self.max_trk_len, val_batch_len, 'validation')

        # Create validation batch
        features = []

        # Get pos track features
        pos_batch = np.zeros((0, 128, 64, 3))
        neg_batch = np.zeros((0, 128, 64, 3))
        for i in range(val_batch_len):
            for j in range(self.max_trk_len-val_trk_len[i], self.max_trk_len):
                np.concatenate((pos_batch, val_pos_img[i, j]), 0)
                np.concatenate((neg_batch, val_neg_img[i, j]), 0)
        pos_feature_batch = self.featureExtractor.predict(pos_batch)
        neg_feature_batch = self.featureExtractor.predict(neg_batch)

        # Create LSTM input batch
        cur_idx = 0
        for i in range(val_batch_len):
            for j in val_trk_len[i]:
                pos_track_features = pos_feature_batch[cur_idx:cur_idx+j]
                neg_track_features = neg_feature_batch[cur_idx:cur_idx+j]

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():

            self.DeepTAMA.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

            for step in range(1, total_epoch+1):
                print("Training step : {}".format(step))
                train_pos_img, train_pos_shp, train_neg_img, train_neg_shp, train_trk_len = dataCls.get_LSTM_batch(self.max_trk_len, 64, 'train')

                # Create training batch
                if train_bb_batch.shape[1] > self.max_sequence_len:
                    self.max_sequence_len = train_bb_batch.shape[1]
                train_features = self.JINet.predict(train_template_batch)

                self.DeepTAMA.fit(train_bb_batch, train_label_batch, batch_size=1, epochs=1, use_multiprocessing=True)

                if step % 100 == 0:
                    val_loss1, acc1 = self.DeepTAMA.evaluate(val_bb_batch, val_label_batch, batch_size=1)
                    tf.summary.scalar('validation loss', val_loss1, step=step)
                    self.DeepTAMA.save(self._save_dir + '/DeepDa-model-{}.h5'.format(step))

    def getJINetLikelihood(self, input_pair):
        likelihood = self.featureExtractor.predict(input_pair)

        return likelihood

    def getFeature(self, input_pair):
        feature = self.featureExtractor(input_pair)

        return feature

    def getLikelihood(self, feature):
        likelihood = self.DeepTAMA.predict(input)

        return likelihood


def main():
    NN = NeuralNet(train_mode='JINet')
    NN.trainJINet()
    """
    NN = NeuralNet(train_mode='LSTM')
    NN.trainLSTM()
    """


if __name__ == "__main__":
    main()