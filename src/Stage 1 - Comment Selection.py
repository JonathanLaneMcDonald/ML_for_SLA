from hashlib import sha256

# comments containing any of these characters will be excluded from the dataset
exclude = {x for x in list(':&')}

# to avoid introducing duplicate comments, hashes of written comments are stored
written = set()

# write comments to files so we can review kept/discarded or train on kept comments
clean = open('japanese comments', 'w', encoding='utf-8')
dirty = open('japanese comments - dirty', 'w', encoding='utf-8')

# comments were saved in the following format
# <0000337.html 005>「県内」といえばやっぱし県内なので、「都内」も都内なのではないでしょうか。
# so we know the comment starts after the first '>'
with open('japanese comments [raw]', 'r', encoding='utf-8') as infile:
	for line in infile:
		start = line.find('>') + 1
		this_line = line[start:]

		# if a comment is 'clean' and hasn't been written, write it and save the hash
		if set(list(this_line)) - exclude == set(list(this_line)):
			comment_hash = sha256(bytes(this_line, encoding='utf-8')).digest()
			if comment_hash in written:
				pass
			else:
				clean.write(this_line)
				written.add(comment_hash)
		# we just want to be able to verify that we're discarding the right stuff
		else:
			dirty.write(this_line)

clean.close()
dirty.close()
