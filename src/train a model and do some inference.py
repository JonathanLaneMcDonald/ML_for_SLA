
import numpy as np
from keras.models import save_model

from statics.ByteNetEncoder import ByteNetEncoder, ByteNetEncoderConfig
from concrete import ContextualEmbeddingsPreTrainingDataGenerator

token_map = {word:int(token) for word, token in [x.split() for x in open('enhanced english comments - dataset bpe token mappings', 'r', encoding='utf-8').read().split('\n') if len(x)]}
print(len(token_map),'items loaded in token mapping')

input_size = 128
training = np.frombuffer(open('enhanced english comments - dataset - 1', 'rb').read(), dtype=np.uint16)
validation = np.frombuffer(open('enhanced english comments - dataset - 2', 'rb').read(), dtype=np.uint16)
print(training.shape, 'comments array shape', len(training))

# training
batch_size = 128
batches = 128
samples = batch_size * batches

model_config = ByteNetEncoderConfig.modified_default(len(token_map), input_size)
model = ByteNetEncoder.get_model(model_config)
training_data_generator = ContextualEmbeddingsPreTrainingDataGenerator.ContextualEmbeddingsPreTrainingDataGenerator(training, token_map, input_size)
validation_data_generator = ContextualEmbeddingsPreTrainingDataGenerator.ContextualEmbeddingsPreTrainingDataGenerator(validation, token_map, input_size)

herstory = []
for e in range(1000):
	train_features, train_positions, train_labels = training_data_generator.generate(samples)
	val_features, val_positions, val_labels = validation_data_generator.generate(samples//10)
	history = model.fit([train_features, train_positions], train_labels, batch_size=batch_size, epochs=1, verbose=1, \
						validation_data=([val_features, val_positions], val_labels))

	herstory.append(history.history['loss'][-1])
	save_model(model, 'enhanced english contextual embeddings loss='+str(herstory[-1])[:6], save_format='h5')
