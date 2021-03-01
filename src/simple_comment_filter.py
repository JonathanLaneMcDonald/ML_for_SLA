
import json
import nltk

from sys import argv

min_length = 1024

size = 0
lines = 0
keepers = 0
clean_texts = open(argv[1]+'_clean','w')
with open(argv[1],'r') as f:
	for line in f:
		size += len(line)
		lines += 1
		
		body = json.loads(line).get('body')

		# i want longer comments
		if min_length < len(body):

			clean_texts.write(body.replace('\r\n','\n').replace('\r','\n').replace('\n','[RET]') + '\n')

			keepers += 1
			if keepers % 10000 == 0:
				print (lines, size//(2**20), keepers)
							
