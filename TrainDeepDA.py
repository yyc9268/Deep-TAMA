from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Concatenate, Conv2D, Flatten, Lambda, MaxPool2D
from tensorflow.keras.backend import sqrt, square, switch, sum, ones_like, print_tensor, constant, maximum, mean, stack, shape, epsilon, squeeze
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import DataSetting as ds
import os


class DeepDa():

    def __init__(self, is_test=False, train_mode='siamese', max_trk_len=5):
        self.max_sequence_len = 0
        self.max_trk_len = max_trk_len

        self._save_dir = 'model'
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

        self._log_dir = 'log'
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

        self.img_input = Input(shape=(128, 64, 3))
        self.feature_dist = Input(shape=(None, 150))
        self.input = Input(shape=(None, 5 * (self.max_trk_len + 1)))

        self.anchor_input = Input(shape=(128, 64, 3))
        self.pos_input = Input(shape=(128, 64, 3))
        self.neg_input = Input(shape=(128, 64, 3))

        if is_test:
            self.lstm_model = load_model(self._save_dir + '/DeepDa-model-{}.h5'.format(1600), custom_objects={'associationLoss': self.associationLoss})
            self.feature_extractor_model = load_model(self._save_dir + '/feature_extractor-model={}.h5'.format(1000), custom_objects={'tripletLoss': self.tripletLoss})
        else:
            if train_mode == 'siamese':
                self.feature_extractor_model = Model(inputs=self.img_input, outputs=self.feature_extractor())
                self.siamese_model = Model(inputs=[self.anchor_input, self.pos_input, self.neg_input], outputs=self.siamese())
                self.lstm_model = []
            else:
                self.feature_extractor_model = load_model(
                    self._save_dir + '/feature_extractor-model={}.h5'.format(1000),
                    custom_objects={'tripletLoss': self.tripletLoss})
                self.lstm_model = Model(inputs=[self.input, self.feature_dist], outputs=self.association_network())

    def feature_extractor(self):
        encode1 = Conv2D(filters=16, kernel_size=9, activation='relu')(self.img_input)
        pool1 = MaxPool2D()(encode1)
        encode2 = Conv2D(filters=32, kernel_size=5, activation='relu')(pool1)
        pool2 = MaxPool2D()(encode2)
        encode3 = Conv2D(filters=64, kernel_size=5, activation='relu')(pool2)
        pool3 = MaxPool2D()(encode3)
        flattened = Flatten()(pool3)
        encode4 = Dense(1150, activation='relu')(flattened)
        feature = Dense(150, activation='relu')(encode4)

        return feature

    def siamese(self):
        anchor_net = self.feature_extractor_model(self.anchor_input)
        pos_net = self.feature_extractor_model(self.pos_input)
        neg_net = self.feature_extractor_model(self.neg_input)

        pos_dist = Lambda(self.euclideanDistance, name='pos_dist')([anchor_net, pos_net])
        neg_dist = Lambda(self.euclideanDistance, name='neg_dist')([anchor_net, neg_net])

        stacked_dists = Lambda(lambda vects: stack(vects, axis=1), name='stacked_dists')([pos_dist, neg_dist])

        return stacked_dists

    def association_network(self):

        encode1 = Dense(64, activation='relu')([self.input, self.feature_dist])
        encode2 = Dense(32, activation='tanh')(encode1)
        forward_lstm = RNN(LSTMCell(128), return_sequences=True, go_backwards=True)(encode2)
        backward_lstm = RNN(LSTMCell(128), return_sequences=True)(encode2)
        concatenated = Concatenate()([forward_lstm, backward_lstm])
        decode1 = Dense(64, activation='relu')(concatenated)
        decode2 = Dense(1, activation='sigmoid')(decode1)

        return decode2

    def associationLoss(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        pos_num = sum(y_true)
        neg_num = sum(ones_like(y_true) - y_true)
        w = neg_num / pos_num
        sqr_diff = square(y_pred - y_true)
        loss = sum(switch(y_true == 1, sqr_diff * w, sqr_diff)) / sum(ones_like(y_true))

        return loss

    def tripletLoss(self, y_true, y_pred):
        loss = mean(maximum(constant(0), y_pred[:,0,0] - y_pred[:,1,0] + constant(2)))
        return loss

    def triplet_accuracy(self, y_true, y_pred):
        return mean(y_pred[:, 0, 0] < y_pred[:, 1, 0] - 1.0)

    def euclideanDistance(self, vecs):
        x, y = vecs
        return sqrt(maximum(sum(square(x-y), axis=1, keepdims=True), epsilon()))

    def trainSiamese(self, total_epoch=1000):
        dataCls = ds.data(check_occlusion=True)
        val_anchor, val_pos, val_neg = dataCls.get_triplet('validation', 64)

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():
            self.siamese_model.compile(loss=self.tripletLoss, optimizer='adam', metrics=[self.triplet_accuracy])

            for step in range(1, total_epoch+1):
                print("Train step : {}".format(step))
                train_anchor, train_pos, train_neg = dataCls.get_triplet('train', 64)
                y_train = np.random.randint(2, size=(1, 2, train_anchor.shape[0])).T

                print(train_anchor.shape)
                print(train_pos.shape)
                print(train_neg.shape)
                print(y_train.shape)

                self.siamese_model.fit([train_anchor, train_pos, train_neg], y_train, batch_size=32, epochs=1)

                if step % 1 == 100:
                    val_loss1, acc1 = self.siamese_model.evaluate([val_anchor, val_pos, val_neg], y_train,
                                                                  batch_size=32)
                    print('{}-step, val_loss1 : {}'.format(step, val_loss1, acc1))
                    tf.summary.scalar('siamese validation loss', val_loss1, step=step)
                    self.siamese_model.save(self._save_dir + '/feature_extractor-model-{}.h5'.format(step))

    def trainLSTM(self, total_epoch=5000):

        dataCls = ds.data()

        val_bb_batch, val_template_batch, val_label_batch = dataCls.get_batch(self.max_trk_len, 1, 'validation')

        summary_writer = tf.summary.create_file_writer('log')
        with summary_writer.as_default():

            self.lstm_model.compile(loss=self.associationLoss, optimizer='adam', metrics=['accuracy'])

            for step in range(1, total_epoch+1):
                print("Training step : {}".format(step))
                train_bb_batch, train_template_batch, train_label_batch = dataCls.get_batch(self.max_trk_len, 1, 'train')

                print('cur len : {}'.format(train_bb_batch.shape[1]))
                print('max len : {}'.format(self.max_sequence_len))
                if train_bb_batch.shape[1] > self.max_sequence_len:
                    self.max_sequence_len = train_bb_batch.shape[1]

                train_features = self.feature_extractor_model.predict(train_template_batch)
                self.lstm_model.fit(train_bb_batch, train_label_batch, batch_size=1, epochs=1, use_multiprocessing=True)

                if step % 100 == 0:
                    val_loss1, acc1 = self.lstm_model.evaluate(val_bb_batch, val_label_batch, batch_size=1)
                    tf.summary.scalar('validation loss', val_loss1, step=step)
                    self.lstm_model.save(self._save_dir + '/DeepDa-model-{}.h5'.format(step))

    def test(self, input):
        assoc_result = self.lstm_model.predict(input)

        return assoc_result


def main():
    deepDa = DeepDa(train_mode='siamese')
    deepDa.trainSiamese()
    # deepDa = DeepDa(train_mode='lstm')
    # deepDa.trainLSTM()


if __name__ == "__main__":
    main()