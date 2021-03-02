
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D
from keras.layers import Add, BatchNormalization, Activation
from keras.layers import GlobalMaxPooling1D, Multiply, SpatialDropout1D
from keras.initializers import RandomNormal
from keras.optimizers import Adam

from concrete.RestorableConfiguredModel import RestorableConfiguredModel

class ByteNetEncoderConfig:

	@staticmethod
	def modified_default(num_tokens, input_size, repr_dimms = 100, hidden_dimms = 600, id_blocks = 4, dilations = [1, 2, 4, 8, 16]):

		return {
			'model_arch': 'ByteNetEncoder',
			'model_type': 'MaskedLanguageModel',
			'num_tokens': num_tokens,
			'input_size': input_size,
			'representation_dimensions': repr_dimms,
			'hidden_dimensions': hidden_dimms,
			'id_blocks': id_blocks,
			'dilations': dilations
		}

class ByteNetEncoder:

	@staticmethod
	def add_projection_layer(x, output_dimms):
		x = Conv1D(filters=output_dimms, kernel_size=1, padding='same', kernel_initializer=RandomNormal(0,0.01))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	@staticmethod
	def add_residual_convolution(x, output_dimms, dilation_rate):
		y = Conv1D(filters=output_dimms, kernel_size=3, dilation_rate=dilation_rate, padding='same', kernel_initializer=RandomNormal(0,0.01))(x)
		y = BatchNormalization()(y)
		x = Add()([x,y])
		x = Activation('relu')(x)
		return x

	@staticmethod
	def get_model(config):

		inputs = Input(shape=(config['input_size'],))
		embeddings = Embedding(config['num_tokens'], config['representation_dimensions'], input_length=config['input_size'])(inputs)

		x = SpatialDropout1D(0.10)(embeddings)

		position = Input(shape=(config['input_size'],1))

		x = ByteNetEncoder.add_projection_layer(x, config['hidden_dimensions'])

		for _ in range(config['id_blocks']):
			for dr in config['dilations']:
				x = ByteNetEncoder.add_residual_convolution(x, config['hidden_dimensions'], dr)

		x = ByteNetEncoder.add_projection_layer(x, config['representation_dimensions'])

		x = Multiply()([x, position])

		x = GlobalMaxPooling1D()(x)

		outputs = Dense(config['num_tokens'], activation='softmax')(x)
		model = Model([inputs, position], outputs)
		model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
		model.summary()

		return RestorableConfiguredModel(model, config)
