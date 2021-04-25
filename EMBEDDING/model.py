import tensorflow as tf
class LSTMClass(tf.keras.Model):
      def __init__(self, **kargs):
    super(LSTMClass, self).__init__()
# 임베딩 층에 현재 단어 사전의 개수에 맞는 input_dim을 설정
# LSTM layer을 거쳐서 특징들을 출력한 이후에
# dense layer2개를 거쳐서 대분류 데이터를 출력을 하게 된다.
# 따라서 output차원의 크기는 1이 된다.

    self.embedding = tf.keras.layers.Embedding(input_dim = kargs['vocab_size'],
                                               output_dim = kargs['embedding_dimension'])
    self.lstm1 = tf.keras.layers.LSTM(units = kargs['lstm_dimension'],
                                      return_sequences = True)
    self.lstm2 = tf.keras.layers.LSTM(units = kargs['lstm_dimension'],
                                      return_sequences = True)
    self.dense1 = tf.keras.layers.Dense(units = kargs['dense_dimension'],
                                       activation = tf.keras.activations.relu,
                                       )
    self.dense2 = tf.keras.layers.Dense(1)
  
  def call(self, x):
    x = self.embedding(x)
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.dense1(x)
    x = self.dense2(x)

    return x

model_name = "LSTMFC2"
batch_size, num_epoch, valid_split = 128, 50, 0.2

kargs = {
    'vocab_size':1340241, # 전체 단어 사전 속 단어의 개수
    'embedding_dimension': 50,
    'lstm_dimension': 150,
    'dense_dimension': 150,
}

model = LSTMClass(**kargs)
model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [tf.keras.metrics.Accuracy(name = 'accuracy')]
)

from keras.callbacks import ModelCheckpoint
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor = 'val_accuracy', verbose = 1,
    save_best_only = True, save_weights_only = True
)
label = tf.cast(df['Large'], tf.int32)
history = model.fit(x = train_inputs, y = train_labels,
                    batch_size = batch_size, epochs = num_epoch, validation_split = valid_split,
                    callbacks = earlystop_callback)