BATCH_SIZE = 512
NUM_EPOCH = 100
VALID_SPLIT = 0.2
MAX_LEN = train_inputs.shape[1]

kwargs = {
    'units' : 128,
    'epoch_size' : NUM_EPOCH,
    'vocab_size' : data_conf['vocab_size'],
    'lstm_dim' : 150,
    'dense_dim' : 150,
    'output_dim' : 21,
    'dropout_rate' : 0.3,
    'embedding_dim' : 50
}

class LSTM(tf.keras.Model):
  def __init__(self, **kwargs):
    super(LSTM, self).__init__(name = MODEL_NAME)
    self.embedding = tf.keras.layers.Embedding(input_dim = kwargs['vocab_size'],
                                               output_dim = kwargs['embedding_dim'])
    self.lstm1 = tf.keras.layers.LSTM(kwargs['lstm_dim'], return_sequences = True)
    self.lstm2 = tf.keras.layers.LSTM(kwargs['lstm_dim'])
    self.dropout = tf.keras.layers.Dropout(kwargs['dropout_rate'])
    self.fc1 = tf.keras.layers.Dense(kwargs['dense_dim'], activation = tf.keras.activations.relu)
    self.fc2 = tf.keras.layers.Dense(kwargs['output_dim'], activation = tf.keras.activations.softmax)

  def call(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.dropout(x)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.fc2(x)

    return x

