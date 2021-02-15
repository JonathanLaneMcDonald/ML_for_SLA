
from concrete import jpTokenizer
from dags.TrainingDataPrepPipeline import TrainingDataPrepPipeline

'''
TrainingDataPrepPipeline(
	jpTokenizer.jpTokenizer(),
	'japanese comments',
	{	'min_chars_for_dict_entry': 10,
		'min_tokens_for_dict_entry': 100,
		'bpe_tokens_to_learn': 30000,
		'max_bpe_tokens_per_doc': 2**14,
		'datablock_write_trigger_size': 2**30,
		'model_input_size': 128
	})
'''

TrainingDataPrepPipeline(
	jpTokenizer.jpTokenizer(),
	'jp comments dev',
	{	'min_chars_for_dict_entry': 1,
		'min_tokens_for_dict_entry': 10,
		'bpe_tokens_to_learn': 5000,
		'max_bpe_tokens_per_doc': 2**14,
		'datablock_write_trigger_size': 2**20,
		'model_input_size': 128
	})

