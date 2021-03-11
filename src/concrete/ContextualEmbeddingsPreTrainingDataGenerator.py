
import numpy as np
from numpy.random import random, choice

from concrete.LinearDataset import LinearDataset
from concrete.SampleBouncer import SampleBouncer
from interfaces.AbstractGenerator import AbstractGenerator
from interfaces.AbstractRestorableObject import AbstractRestorableObject


class ContextualEmbeddingsPreTrainingDataGenerator(AbstractGenerator, AbstractRestorableObject):

	@staticmethod
	def display_state(state, reverse_map):
		for token in state:
			print(reverse_map[token], end=' ')
		print()

	def __init__(self, dataset, token_map, input_size, max_blast_radius = 5):
		self.dataset = dataset
		self.forward_map = token_map
		self.reverse_map = {token:word for word, token in self.forward_map.items()}
		self.model_input_size = input_size
		self.sample_bouncer = SampleBouncer()
		self.max_blast_radius = max_blast_radius

	def get_token_count(self):
		return len(self.forward_map)

	def get_state(self):
		return {
			# needs to be specified as a LinearDataset because these descriptors won't necessarily work with other dataset types
			'dataset_descriptors': self.dataset.get_state(),
			'forward_map': self.forward_map,
			'model_input_size': self.model_input_size,
			'sample_bouncer_descriptors': self.sample_bouncer.get_state(),
			'max_blast_radius': self.max_blast_radius
		}

	@staticmethod
	def restore_state(state: dict):
		try:
			linear_dataset = LinearDataset.restore_state(state['dataset_descriptors'])
			forward_map = state['forward_map']
			model_input_size = state['model_input_size']
			sample_bouncer = SampleBouncer.restore_state(state['sample_bouncer_descriptors'])
			max_blast_radius = state['max_blast_radius']

			data_generator = ContextualEmbeddingsPreTrainingDataGenerator(linear_dataset, forward_map, model_input_size, max_blast_radius)

			data_generator.sample_bouncer = sample_bouncer

			return data_generator

		except Exception as e:
			raise Exception("ContextualEmbeddingsPreTrainingDataGenerator::restore_state(): ", e)

	def generate(self, samples, is_for_validation=False):
		features = np.zeros((samples, self.model_input_size), dtype=np.uint16) + self.forward_map['[PAD]']
		positions = np.zeros((samples, self.model_input_size, 1), dtype=np.uint8)
		labels = np.zeros((samples, 1), dtype=np.uint16)

		s = 0
		spans = dict()
		resamples = 0
		while s < samples:
			epicenter = self.dataset.random_training_position()
			if is_for_validation:
				epicenter = self.dataset.random_validation_position()
			blast_radius = int(random()*self.max_blast_radius)
			crater = {x for x in range(epicenter, epicenter+blast_radius+1)}
			knockout = choice(list(crater))

			# a random token has been selected for knockout, test if we need to resample based on value or sample frequency
			if 0 <= knockout < self.dataset.get_dataset_size() and \
					self.dataset[knockout] != self.forward_map['[SEG]'] and \
					self.sample_bouncer.accept_token(self.dataset[knockout]):

				self.sample_bouncer.register_sampled(self.dataset[knockout])

				# figure out how far from the start of the document we are
				start = knockout - 1
				while self.dataset[start] != self.forward_map['[SEG]'] and knockout - start < self.model_input_size:
					start -= 1

				# figure out how far from the end of the document we are
				finish = knockout + 1
				while self.dataset[finish] != self.forward_map['[SEG]'] and finish - knockout < self.model_input_size:
					finish += 1

				# if the document already fits in the model, then we don't need to do anything
				if finish - start < self.model_input_size:
					pass
				# otherwise, advance the start point and pick a finish position such that the input still fits
				else:
					start += int(random()*(knockout - start))
					finish = min(finish, start + self.model_input_size - 1)

				for i in range((finish+1) - start):
					if ((start+i) in crater) and (self.dataset[start+i] != self.forward_map['[SEG]']):
						features[s][i] = self.forward_map['[MASK]']
						if start+i == knockout:
							positions[s][i][0] = 1
							labels[s][0] = self.dataset[start+i]
					else:
						features[s][i] = self.dataset[start+i]

				if int(blast_radius) not in spans:
					spans[int(blast_radius)] = 0
				spans[int(blast_radius)] += 1

				s += 1
			else:
				resamples += 1

		print(samples, resamples, self.sample_bouncer.get_token_register_size(), sorted(spans.items()))
		ContextualEmbeddingsPreTrainingDataGenerator.display_state(features[-1], self.reverse_map)

		return features, positions, labels
