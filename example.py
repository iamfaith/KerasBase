import functools
from keras import callbacks
class Decorator(object):
    @staticmethod
    def log(is_debug=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                if is_debug:
                    print('call %s():' % func)
                    print('parameters:', args, kw)
                return func(*args, **kw)

            return wrapper

        return decorator

    # @staticmethod
    # def test(decorated):
    #     pass


class BaseTrainer(object):
    __suffix = ".hdf5"
    __checkpoint = "{epoch:02d}-checkpoint" + __suffix
    __cp_folder = os.path.abspath(os.getcwd()) + "/checkpoint"
    cp_path = __cp_folder + "/" + __checkpoint
    model_path = "./model"
    __log_name = "./training_analysis.csv"

    def __init__(self):
        self.has_train = False
        print(BaseTrainer.__cp_folder)
        files = []
        if os.path.exists(BaseTrainer.__cp_folder):
            for f in os.listdir(BaseTrainer.__cp_folder):
                if f.endswith(BaseTrainer.__suffix):
                    self.has_train = True
                    files.append(f)
        else:
            os.makedirs(BaseTrainer.__cp_folder)
        files.sort(key=lambda f: f, reverse=True)
        # self.csv_logger = CSVLogger(BaseTrainer.__log_name, separator=',', append=False)
        self.checkpointer = callbacks.ModelCheckpoint(filepath=BaseTrainer.cp_path, verbose=1, save_best_only=True,
                                                      monitor='loss')
        if self.has_train:
            from keras.models import load_model
            self.cp_file = files[0]
            self.epoch = int(self.cp_file.split('-')[0])
            print("Load checkpoint:", files)
            self.model = tf.keras.models.load_model(BaseTrainer.__cp_folder + "/" + files[0])











from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import keras
class CNN(BaseTrainer):

  @Decorator.log(True)
  def load_data(self, **dict):
        print(dict)
        self.X_train = dict['train_x']
        self.y_train = dict['train_y']
        self.X_test = dict['test_x']
        self.y_test = dict['test_y']

  def train(self):
    batch_size = 128
    epochs = 100
    num_classes = 11

    input_shape = (128, 128, 3)

    if self.has_train:
            nb_epoch = nb_epoch - self.epoch
            print('new epoch', nb_epoch)
            self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs,
                           callbacks=[self.checkpointer],  verbose=1)
    else:

      self.model = Sequential()
      # self.model.add(Conv2D(32, kernel_size=(3, 3),
      #                 activation='relu',
      #                 input_shape=input_shape))
      # self.model.add(Conv2D(64, (3, 3), activation='relu'))
      # self.model.add(MaxPooling2D(pool_size=(2, 2)))
      # self.model.add(Dropout(0.25))
      # self.model.add(Dropout(0.25))
      # self.model.add(Flatten())
      # self.model.add(Dense(128, activation='relu'))
      # self.model.add(Dropout(0.5))
      # self.model.add(Dense(num_classes, activation='softmax'))



      self.model.add(Conv2D(64, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=input_shape))
      self.model.add(BatchNormalization())
      self.model.add(MaxPooling2D(pool_size=(2, 2)))




      self.model.add(Conv2D(128, (3, 3), activation='relu'))
      self.model.add(BatchNormalization())
      self.model.add(MaxPooling2D(pool_size=(2, 2)))

      self.model.add(Conv2D(256, (3, 3), activation='relu'))
      self.model.add(BatchNormalization())
      self.model.add(MaxPooling2D(pool_size=(2, 2)))


      self.model.add(Conv2D(512, (3, 3), activation='relu'))
      self.model.add(BatchNormalization())
      self.model.add(MaxPooling2D(pool_size=(2, 2)))

      # self.model.add(Flatten())
      # self.model.add(MaxPooling2D(pool_size=(2, 2)))
      # self.model.add(Dropout(0.25))
      self.model.add(Flatten())
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(num_classes, activation='softmax'))



      print(self.model.summary())
      self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer='adam',
                    metrics=['accuracy'])

      self.model.fit(self.X_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[self.checkpointer],
                verbose=1)
      self.model.save("./cnn_model.hdf5")
      score = self.model.evaluate(self.X_test, self.y_test, verbose=1)
      print('Test loss:', score[0])
      print('Test accuracy:', score[1])

  def predict(self):
    pass
