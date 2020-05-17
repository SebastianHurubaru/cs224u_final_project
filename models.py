import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_model(model_name):

    if model_name == 'baseline':
        return BaselineModel

class BaselineModel(Model):

    def __init__(self, args, **kwargs):
        super(BaselineModel, self).__init__(kwargs)

        self.args = args

        self.out = Dense(2)

    def call(self, x):

        out = self.out(tf.reshape(x, [-1, 1]))

        return out