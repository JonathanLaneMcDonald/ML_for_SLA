
"""
Training Regime:
	Train on 500 million samples
	Fine-tune on 100 million samples

Output:
	Write the strongest and weakest 100 examples of context for every word between x and y percentile by frequency (probably 80-99%)

"""

import numpy as np

charmap, rlut, fmap = dict(), dict(), dict()
with open('japanese comments dataset supporting info','r',encoding='utf-8') as f:
	for line in f:
		v = line[:-1].split('\t')
		if len(v) == 3:
			key, value, counts = v[0], int(v[1]), int(v[2])
			charmap[key] = value
			rlut[value] = key
			fmap[value] = counts

rlut[0xffff] = '[SEG]'
rlut[0xfffe] = '[MASK]'

print (len(charmap),'items loaded in charmap',v)

input_size = 128
comments = np.frombuffer(open('japanese comments dataset','rb').read(),dtype=np.uint16)
print (comments.shape,'comments array shape',len(comments))

from numpy.random import random as npr

from keras.models import Model, load_model, save_model
from keras.layers import Input, Embedding, Dense, Conv1D
from keras.layers import Add, BatchNormalization, Activation
from keras.layers import GlobalMaxPooling1D, Multiply, SpatialDropout1D
from keras.initializers import RandomNormal
from keras.optimizers import Adam

def build_model(dimms, blocks, dilations):

	inputs = Input(shape=(input_size,))
	embeddings = Embedding(2**16, dimms, input_length=input_size)(inputs)

	x = SpatialDropout1D(0.1)(embeddings)

	position = Input(shape=(input_size,1))

	for _ in range(blocks):
		for dr in dilations:
			y = Conv1D(filters=dimms, kernel_size=3, dilation_rate=dr, padding='same', kernel_initializer=RandomNormal(0,0.01))(x)
			y = BatchNormalization()(y)
			y = Activation('relu')(y)
			x = Add()([x,y])

	x = Multiply()([x, position])

	x = GlobalMaxPooling1D()(x)

	outputs = Dense(2**16, activation='softmax')(x)
	return Model([inputs, position], outputs)

def measure_area_under_curve(model, samples, dataset, input_size, fmap, sample_range, resample_frequency):
	"""here, i'm going to build a test dataset and measure the model's accuracy at filling in the blank"""

	d = 1000
	features = np.zeros((d, input_size),dtype=np.uint16) + 0xffff
	positions = np.zeros((d, input_size, 1),dtype=np.uint8)
	labels = np.zeros((d, 1),dtype=np.uint16)

	s = 0
	auc = 0
	ppl = 0
	resamples = 0
	while s < samples:
		ko = sample_range[0] + int(npr()*(sample_range[1]-input_size))
		if dataset[ko] != 0xffff and npr()*fmap[dataset[ko]] < resample_frequency:
			start = ko - 1
			while 0 < start and ko - start < input_size and dataset[start]:
				start -= 1

			start = ko - min(int(npr()*input_size), ko - start)

			finish = ko + 1
			while finish < len(dataset) and finish - start < input_size and dataset[finish]:
				finish += 1

			for i in range(finish - start):
				if start+i == ko:
					features[s%d][i] = 0xfffe
					positions[s%d][i][0] = 1
					labels[s%d][0] = dataset[start+i]
				else:
					features[s%d][i] = dataset[start+i]
	
			s += 1

			if s and s % d == 0:
				predictions = model.predict([features, positions])

				for p in range(len(predictions)):
					auc += predictions[p][labels[p][0]]
					ppl += np.log(max(predictions[p][labels[p][0]], 10**-9))/np.log(2)

				if s % 10000 == 0:
					print (s,resamples,auc/s,2**(-ppl/s))

				features = np.zeros((d, input_size),dtype=np.uint16)
				positions = np.zeros((d, input_size, 1),dtype=np.uint8)
				labels = np.zeros((d, 1),dtype=np.uint16)

		else:
			resamples += 1

	return auc/samples, 2**(-ppl/samples)

def score_examples(model, dataset, input_size, fmap, threshold=100):
	print ('threshold:',threshold)

	whitelist = set()
	blacklist = set()
	sum_total = sum([counts for key, counts in fmap.items() if key >= 2])
	cumulative = 0
	for counts, key in list(reversed(sorted([(counts, key) for key, counts in fmap.items() if key >= 2]))):
		cumulative += counts
		if cumulative/sum_total < 0.85 or cumulative/sum_total > 0.995:
			blacklist.add(key)
		else:
			whitelist.add(key)
	print (len(blacklist),'items added to blacklist')
	print (len(whitelist),'items added to whitelist')

	d = 10000
	features = np.zeros((d, input_size),dtype=np.uint16) + 0xffff
	positions = np.zeros((d, input_size, 1),dtype=np.uint8)
	labels = np.zeros((d, 1),dtype=np.uint16)

	positive = dict()
	negative = dict()

	s = 0

	start = 0
	finish = start + 1
	while finish < len(dataset) and dataset[finish] != 0xffff:
		finish += 1

	for ko in range(len(dataset)):
		if dataset[ko] != 0xffff:
			if finish - start < input_size and dataset[ko] not in blacklist:
				for i in range(finish - start):
					if start+i == ko:
						features[s%d][i] = 0xfffe
						positions[s%d][i][0] = 1
						labels[s%d][0] = dataset[start+i]
					else:
						features[s%d][i] = dataset[start+i]

				s += 1
		
				if s and s % d == 0:
					print (ko, s)
					predictions = model.predict([features, positions])

					for p in range(len(predictions)):
						if labels[p][0] not in positive:
							positive[labels[p][0]] = []

						if labels[p][0] not in negative:
							negative[labels[p][0]] = []

						question = [rlut[features[p][x]] for x in range(len(features[p]))]
						answer = rlut[labels[p][0]]

						positive[labels[p][0]].append((predictions[p][labels[p][0]], (question, answer)))
						if len(positive[labels[p][0]]) > threshold:
							positive[labels[p][0]] = list(reversed(sorted(positive[labels[p][0]])))[:threshold]
							
						negative[labels[p][0]].append((predictions[p][labels[p][0]], (question, answer)))
						if len(negative[labels[p][0]]) > threshold:
							negative[labels[p][0]] = sorted(negative[labels[p][0]])[:threshold]

					if s % 1000000 == 0:
						# write everything out to a file
						pe = open('japanese comments with strong contexts','w')
						for key, values in positive.items():
							for value in values:
								pe.write(value[1][1] + '\t' + ' '.join(value[1][0]) + '\n')
						pe.close()

						ne = open('japanese comments with weak contexts','w')
						for key, values in negative.items():
							for value in values:
								ne.write(value[1][1] + '\t' + ' '.join(value[1][0]) + '\n')
						ne.close()

					features = np.zeros((d, input_size),dtype=np.uint16) + 0xffff
					positions = np.zeros((d, input_size, 1),dtype=np.uint8)
					labels = np.zeros((d, 1),dtype=np.uint16)
		else:
			start = ko
			finish = start + 1
			while finish < len(dataset) and dataset[finish] != 0xffff:
				finish += 1

def generate_dataset(samples, dataset, input_size, fmap, sample_range, resample_frequency):
	features = np.zeros((samples, input_size),dtype=np.uint16) + 0xffff
	positions = np.zeros((samples, input_size, 1),dtype=np.uint8)
	labels = np.zeros((samples, 1),dtype=np.uint16)

	s = 0
	words = set()
	resamples = 0
	ko_offsets = {index:0 for index in range(input_size)}
	while s < samples:
		ko = sample_range[0] + int(npr()*(sample_range[1]-input_size))
		if dataset[ko] != 0xffff and npr()*fmap[dataset[ko]] < resample_frequency:
			start = ko - 1
			while 0 < start and ko - start < input_size and dataset[start]:
				start -= 1

			start = ko - min(int(npr()*input_size), ko - start)

			ko_offsets[ko - start] += 1

			finish = ko + 1
			while finish < len(dataset) and finish - start < input_size and dataset[finish]:
				finish += 1

			for i in range(finish - start):
				if start+i == ko:
					features[s][i] = 0xfffe
					positions[s][i][0] = 1
					labels[s][0] = dataset[start+i]
					words.add(dataset[start+i])
				else:
					features[s][i] = dataset[start+i]

			s += 1

			if s % 100000 == 0:
				print (s, resamples, len(words))
		else:
			resamples += 1

	print (resamples, 'resamples', len(words), 'unique words sampled')
	print (ko_offsets)

	return features, positions, labels

enfranchisement_threshold = 10000

nonzero_occurrences = list(reversed(sorted([value for key, value in fmap.items() if value])))
resample_frequency = nonzero_occurrences[enfranchisement_threshold]
print ('resample frequency:',resample_frequency,'evenly samples',int(enfranchisement_threshold))

area_under_curve = []
perplexity = []

validation_split = 0.1
train_sample_start_index = input_size
train_sample_end_index = int(len(comments)*(1-validation_split)) - train_sample_start_index
validation_sample_start_index = train_sample_end_index
validation_sample_end_index = len(comments) - validation_sample_start_index - input_size

train_range = (train_sample_start_index, train_sample_end_index)
validation_range = (validation_sample_start_index, validation_sample_end_index)

print ('ranges',train_range,validation_range)

def million(n):
	return n*1000*1000

def train_model(model, samples, epochs, batch_size, area_under_curve, perplexity):

	for e in range(epochs):

		auc, ppl = measure_area_under_curve(model, 100000, comments, input_size, fmap, validation_range, resample_frequency)
		area_under_curve += [auc]
		perplexity += [ppl]
		print ('auc',', '.join([str(x)[:6] for x in area_under_curve]))
		print ('ppl',', '.join([str(int(x)) for x in perplexity]))

		if np.argmax(area_under_curve) == len(area_under_curve)-1:
			save_model(model, 'jp dae auc='+str(area_under_curve[-1])[:6]+' ppl='+str(int(perplexity[-1])), save_format='h5')
			print ('model saved')

		features, positions, labels = generate_dataset(samples, comments, input_size, fmap, train_range, resample_frequency)
		model.fit([features, positions],labels,batch_size=batch_size,epochs=1,verbose=1)

# training
epochs = 50
batch_size = 256
batches = 10000
samples = batch_size * batches

#model = build_model(600, 4, [1,2,4,8])

model = load_model('jp dae uniform across 10k auc=0.0760 ppl=356')
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model.summary()
#train_model(model, samples, epochs, batch_size, area_under_curve, perplexity)

# inference
score_examples(model, comments, input_size, fmap)

#model = load_model('jp dae auc=0.1722 ppl=105')
#model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.0001))
#model.summary()
