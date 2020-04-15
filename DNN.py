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
        nb_epoch = 100
        if self.has_train:
            nb_epoch = nb_epoch - self.epoch
            print('new epoch', nb_epoch)
            self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epoch,
                           callbacks=[self.checkpointer, self.csv_logger])
        else:
            # 1. define the network
            self.model = Sequential()
            self.model.add(Dense(1024, input_dim=41, activation='relu'))
            self.model.add(Dropout(0.01))
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epoch,
                      callbacks=[self.checkpointer, self.csv_logger])
            self.model.save("./dnn1layer_model.hdf5")
        score, acc = self.model.evaluate(self.X_test, self.y_test)    
        print('Test score:', score)
        print('Test accuracy', acc)


if __name__ == '__main__':
    dnn = DNN()
    dnn.load_data(train='./kdd/binary/Training.csv', test='./kdd/binary/Testing.csv')
    dnn.train()
