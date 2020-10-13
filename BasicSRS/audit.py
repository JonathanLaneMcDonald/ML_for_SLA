
"""
Read the tokens.db file and write a very simple progress report to stdout
"""

import numpy as np

vocab = set()

activations = np.zeros(1000 * 365)
maintenanceReps = np.zeros(1000 * 365)
futureReps = np.zeros(1000 * 365)
for key, repped, toBeRepped in [x.split() for x in open('tokens.db', 'r', encoding='utf-8').read().split('\n') if len(x)]:
	if key in vocab:
		maintenanceReps[int(repped) // 86400] += 1
	else:
		activations[int(repped) // 86400] += 1
		vocab.add(key)
	futureReps[int(toBeRepped) // 86400] += 1

start = 0
while activations[start] + maintenanceReps[start] + futureReps[start] == 0:
	start += 1

finish = len(activations) - 1
while activations[finish] + maintenanceReps[finish] + futureReps[finish] == 0:
	finish -= 1

print(',activations,maintenance,scheduled')
for i in range(start, finish + 1):
	print(activations[i], ',', maintenanceReps[i], ',', futureReps[i])
