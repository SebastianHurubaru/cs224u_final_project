import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import layers

from transformers import TFBertModel

def create_model(model_name):

    if model_name == 'baseline':
        return BaselineModel
    elif model_name == 'lstm':
        return LSTMModel

class LSTMModel(Model):

    def __init__(self, args, **kwargs):
        super(LSTMModel, self).__init__(kwargs)

        self.args = args

        self.rnn = LSTM(self.args.hidden_size)

        self.out = Dense(2)

        self.dropout = Dropout(self.args.drop_prob)

    def call(self, x):
        x = tf.dtypes.cast(x, tf.float32)
        batch_size = x.shape[0]
        print("x.shape: {}".format(x.shape))

        sent_emb_reshaped = tf.reshape(x, [batch_size, -1, 512])

        print("sent_emb_reshaped.shape: {}".format(sent_emb_reshaped.shape))

        h = self.dropout(self.rnn(sent_emb_reshaped))

        print("h.shape: {}".format(h.shape))

        out = self.out(h)

        return out

class BaselineModel(Model):

    def __init__(self, args, **kwargs):
        super(BaselineModel, self).__init__(kwargs)

        self.args = args

        # Load pre-trained model (weights)
        bert_version = 'bert-large-cased-whole-word-masking'
        self.bert_model = TFBertModel.from_pretrained(bert_version)

        self.rnn = LSTM(self.args.hidden_size)

        self.out = Dense(2)

        self.dropout = Dropout(self.args.drop_prob)

    def call(self, x):

        batch_size = x.shape[0]

        x = tf.reshape(x, [-1, self.args.max_sentence_length])

        # Create the attention mask
        att_mask = tf.ones_like(x).numpy()
        att_mask[(x == 0).numpy()] = 0
        att_mask = tf.convert_to_tensor(att_mask)

        # Get the context pre-trained embeddings
        context_embeddings = self.dropout(self.bert_model(x, attention_mask=att_mask)[0])

        # Do an average pooling of the word embeddings to get the sentence embeddings
        sentence_embeddings = tf.reduce_mean(context_embeddings, axis=-2)
        sent_emb_reshaped = tf.reshape(sentence_embeddings, [batch_size, -1, self.bert_model.config.max_length])
        h = self.dropout(self.rnn(sent_emb_reshaped))

        out = self.out(h)

        return out