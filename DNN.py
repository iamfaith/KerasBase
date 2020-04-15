import pandas as pd
import numpy as np
from common import *

np.random.seed(1337)  # for reproducibility
from sklearn.preprocessing import Normalizer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Dropout, Activation, Embedding
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class DNN(BaseTrainer):

    @Decorator.log(True)
    def load_data(self, **dict):
        print(dict)
        self.traindata = pd.read_csv(dict['train'], header=None)
        self.testdata = pd.read_csv(dict['test'], header=None)
        X = self.traindata.iloc[:, 1:42]
        Y = self.traindata.iloc[:, 0]
        C = self.testdata.iloc[:, 0]
        T = self.testdata.iloc[:, 1:42]
        trainX = np.array(X)
        testT = np.array(T)
        trainX.astype(float)
        testT.astype(float)
        scaler = Normalizer().fit(trainX)
        trainX = scaler.transform(trainX)
        scaler = Normalizer().fit(testT)
        testT = scaler.transform(testT)

        self.y_train = np.array(Y)
        self.y_test = np.array(C)

        self.X_train = np.array(trainX)
        self.X_test = np.array(testT)

    def train(self):
        batch_size = 64
        nb_epoch = 105
        if self.has_train:
            nb_epoch = nb_epoch - self.epoch
            print('new epoch', nb_epoch)
            self.model.fit(self.X_train, self.y_train, epochs=batch_size, batch_size=batch_size,
                           callbacks=[self.checkpointer, self.csv_logger])
        else:
            # 1. define the network
            model = Sequential()
            model.add(Dense(1024, input_dim=41, activation='relu'))
            model.add(Dropout(0.01))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(self.X_train, self.y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      callbacks=[self.checkpointer, self.csv_logger])
            model.save("./dnn1layer_model.hdf5")


if __name__ == '__main__':
    dnn = DNN()
    dnn.load_data(train='../kdd/binary/Training.csv', test='../kdd/binary/Testing.csv')
    dnn.train()
