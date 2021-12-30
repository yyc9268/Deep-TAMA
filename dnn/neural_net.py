import os
import random

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization,\
    Activation, Softmax, Bidirectional, LSTM, Dropout
from tensorflow.keras.utils import plot_model
import numpy as np


class neuralNet:
    def __init__(self, is_test=False, continue_epoch=0, train_mode='lstm', config=None):
        """
        Initialize neuralNet class
        :param is_test: test_mode On/Off
        :param continue_epoch: continue from existing checkpoint epoch
        :param train_mode: 'lstm' or 'jinet'
        :param config: config class
        """
        self.max_sequence_len = 0
        self.max_trk_len = config.lstm_len
        self.matching_feature_sz = 150

        self._save_dir = config.model_dir
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

        self._log_dir = config.log_dir
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

        self.img_input = Input(shape=(128, 64, 6))
        self.lstm_input = Input(shape=(self.max_trk_len, self.matching_feature_sz+3))

        model_names = {'lstm', 'jinet'}
        self.nets = {'lstm': Model(inputs=self.lstm_input, outputs=self.deeptama_output()),
                     'jinet': Model(inputs=self.img_input, outputs=self.jinet_output())}

        if is_test:
            for model_name in model_names:
                self.load_model(model_name, config.model[model_name]['save_name'], config.model[model_name]['tot_epoch'])
        else:
            if train_mode == 'jinet':
                if continue_epoch > 0:
                    self.load_model('jinet', config.model['jinet']['save_name'], continue_epoch)
            elif train_mode == 'lstm':
                self.load_model('jinet', config.model['jinet']['save_name'], config.model['jinet']['tot_epoch'])
                if continue_epoch > 0:
                    self.load_model('lstm', config.model['lstm']['save_name'], continue_epoch)
            else:
                raise NotImplementedError

        self.featureExtractor = Model(inputs=self.nets['jinet'].inputs,
                                      outputs=self.nets['jinet'].get_layer('matching_feature').output)

    def __del__(self):
        print("neural-net deleted")

    def load_model(self, model_name, save_name, epoch):
        """
        Load model
        """
        self.nets[model_name] = load_model(self._save_dir + '/{}-{}.h5'.format(save_name, epoch))
        print('Loaded : ', self._save_dir + '/{}-{}.h5'.format(save_name, epoch))

    def plot_model(self, model_name=''):
        """
        Plot model shape
        :param model_name: 'lstm' or 'jinet'
        :return: None
        """
        model_name = model_name.lower()
        model_save_dir = os.path.join(self._log_dir, '{}.png'.format(model_name))
        plot_model(self.nets[model_name], to_file=model_save_dir, show_shapes=True)

    def jinet_output(self):
        """
        JI-Net network structure
        :return: likelihood of input pair
        """
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
        drop1 = Dropout(rate=0.2)(flattened)
        encode4 = Dense(1152)(drop1)
        bnorm4 = BatchNormalization()(encode4)
        relu4 = Activation('relu')(bnorm4)
        encode5 = Dense(self.matching_feature_sz)(relu4)
        bnorm5 = BatchNormalization()(encode5)
        relu5 = Activation('relu', name='matching_feature')(bnorm5)
        logit = Dense(2)(relu5)
        likelihood = Softmax()(logit)

        return likelihood

    def deeptama_output(self):
        """
        LSTM network structure (The implementation is slightly different from the paper)
        :return: likelihood of input sequence
        """
        encode1 = Dense(153)(self.lstm_input)  # 153 -> 153
        bnorm1 = BatchNormalization()(encode1)
        tanh1 = Activation('tanh')(bnorm1)
        lstm_out = RNN(LSTMCell(128), return_sequences=False, go_backwards=False)(tanh1)
        decode1 = Dense(64)(lstm_out)
        bnorm2 = BatchNormalization()(decode1)
        relu1 = Activation('relu')(bnorm2)
        logit = Dense(2)(relu1)
        likelihood = Softmax()(logit)

        return likelihood

    def create_lstm_input(self, img_batch, shp_batch, trk_len, is_test=False):
        """
        Create LSTM input from sequence of images and shapes.
        :param img_batch: (N, 6, 128, 64, 3)
        :param shp_batch: (N, 6, 3)
        :param trk_len: Valid length of each track sequence (N)
        :return: LSTM input batch (N, 6, 153)
        """
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
        if not is_test:
            random.shuffle(shuffled_idx)

        return input_batch, shuffled_idx

    def get_jinet_likelihood(self, input_pair):
        """
        Get appearance likelihood form input_pair
        :param input_pair: numpy array of shape (N, 128, 64, 6)
        :return: likelihood batch
        """
        likelihood = self.featureExtractor.predict(input_pair)

        return likelihood

    def get_feature(self, input_pair):
        """
        Get matching feature from input_pair
        :param input_pair: numpy array of shape (N, 128, 64, 6)
        :return: matching feature batch (N, 150)
        """
        feature = self.featureExtractor(input_pair)

        return feature

    def get_likelihood(self, lstm_input):
        """
        Get appearance likelihood from JI-Net embedded features
        :param lstm_input: (N, 6, 152)
        :return: likelihood batch
        """
        likelihood = self.nets['lstm'].predict(lstm_input)

        return likelihood
