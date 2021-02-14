
from concrete import jpTokenizer
from dags.TrainingDataPrepPipeline import TrainingDataPrepPipeline

'''
config = dict()
config['min_chars_for_dict_entry'] = 10
config['min_tokens_for_dict_entry'] = 100
config['bpe_tokens_to_learn'] = 30000
config['max_bpe_tokens_per_doc'] = 256
config['datablock_write_trigger_size'] = 2**30

TrainingDataPrepPipeline.TrainingDataPrepPipeline.execute(
	jpTokenizer.jpTokenizer(),
	'japanese comments',
	config
)
'''

config = dict()
config['min_chars_for_dict_entry'] = 1
config['min_tokens_for_dict_entry'] = 10
config['bpe_tokens_to_learn'] = 5000
config['max_bpe_tokens_per_doc'] = 256
config['datablock_write_trigger_size'] = 2**30

TrainingDataPrepPipeline(
	jpTokenizer.jpTokenizer(),
	'jp comments dev',
	config)

