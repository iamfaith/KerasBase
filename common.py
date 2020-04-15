import os
import functools
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
import tensorflow as tf


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
        self.csv_logger = CSVLogger(BaseTrainer.__log_name, separator=',', append=False)
        self.checkpointer = callbacks.ModelCheckpoint(filepath=BaseTrainer.cp_path, verbose=1, save_best_only=True,
                                                      monitor='loss')
        if self.has_train:
            from keras.models import load_model
            print(files)
            self.model = tf.keras.models.load_model(BaseTrainer.__cp_folder + "/" + files[0])


if __name__ == '__main__':
    base = BaseTrainer()
