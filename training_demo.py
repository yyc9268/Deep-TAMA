from dnn.neural_net import neuralNet


def main():
    # Train JI-Net first
    #NN = neuralNet(train_mode='JINet')
    #NN.train_jinet()
    #del NN

    # Train LSTM using pre-trained JI-Net
    NN = neuralNet(train_mode='LSTM')
    NN.train_lstm()
    del NN


if __name__ == "__main__":
    main()
