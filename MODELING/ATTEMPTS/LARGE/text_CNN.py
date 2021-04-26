model_name = 'CNN_Classifier01'
BATCH_SIZE = 512
NUM_EPOCH = 30
VALID_SPLIT = 0.3
MAX_LEN = train_inputs.shape[1]

kargs = {'model_name' : model_name,
         'vocab_size' : data_configs['vocab_size'],
         'embedding_size' : 128,
         'num_filters' : 100,
         'dropout_rate' : 0.3,
         'hidden_dimension' : 250,
         'output_dimension' : 1}      

class CNNClassifier(tf.keras.Model):
      def __init__(self, **kargs):
    super(CNNClassifier, self).__init__(name = kargs['model_name'])
    self.embedding = layers.Embedding(input_dim = kargs['vocab_size'],
                                      output_dim = kargs['embedding_size'],
                                      )
    self.conv_list = [layers.Conv1D(filters = kargs['num_filters'], kernel_size = kernel_size,
                                    strides = 1, padding = 'valid',
                                    activation = tf.keras.activations.relu, 
                                    kernel_constraint = tf.keras.constraints.MaxNorm(max_value=3.))
    for kernel_size in [3,4,5]]
    self.pooling = layers.GlobalMaxPooling1D()
    self.dropout = layers.Dropout(kargs['dropout_rate'])

    self.fc1 = layers.Dense(units = kargs['hidden_dimension'],
                                          activation = tf.keras.activations.relu,
                                          kernel_constraint = tf.keras.constraints.MaxNorm(max_value = 3.))
    self.fc2 = layers.Dense(units = kargs['output_dimension'],
                            activation = tf.keras.activations.softmax,
                            kernel_constraint = tf.keras.constraints.MaxNorm(max_value = 3.))
  
  def call(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis = 1)
    x = self.fc1(x)
    x = self.fc2(x)

    return x


model = CNNClassifier(**kargs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [tf.keras.metrics.CategoricalAccuracy(name = 'accuracy')]
)

earlystop_callback = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.001, patience = 3)

history = model.fit(train_inputs, train_labels, batch_size = BATCH_SIZE, epochs = NUM_EPOCH,
                    validation_split = VALID_SPLIT, callbacks = [earlystop_callback, cp_callback])