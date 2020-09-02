'''
the main purpose is to tokenize and label documents using a dictionary of terms
'''

import numpy

class TreeNode:
	def __init__(self):
		self.children = dict()
		self.isTerminal = False

class Tree:
	"""Represents sequences in a tree for fast string matching"""

	def __init__(self, arrayOfTokenized = []):
		self.root = TreeNode()
		self.longest_key = 0

		for tokenized in arrayOfTokenized:
			self.registerTokenized(tokenized)

	def getMaxKeyLength(self):
		return self.longest_key

	def registerTokenized(self, tokenized):
		"""Add a sequence to the tree, adding leaf nodes where necessary"""

		if self.longest_key < len(tokenized):
			self.longest_key = len(tokenized)

		pNode = self.root
		for t in range(len(tokenized)):
			if not tokenized[t] in pNode.children:
				pNode.children[tokenized[t]] = TreeNode()
			pNode = pNode.children[tokenized[t]]
		pNode.isTerminal = True

	def getLongestMatch(self, tokenized):
		"""Find the provided sequence using the existing tree"""

		pNode = self.root
		longest_match = 0
		for t in range(len(tokenized)):
			if not tokenized[t] in pNode.children:
				return longest_match
			else:
				pNode = pNode.children[tokenized[t]]
				if pNode.isTerminal:
					longest_match = t+1
		return longest_match

class SequenceTagger:
	"""Provide higher level functionality by wrapping the Tree (Trie) datastructure"""

	def __init__(self, arrayOfTokenized):
		if len(arrayOfTokenized) == 0:
			raise ValueError('Dictionary must not be empty')

		self.tree = Tree(arrayOfTokenized)

	def getMaxKeyLength(self):
		return self.tree.getMaxKeyLength()

	def tagDocument(self, tokenizedDoc, boolean_result = False, return_labels = False):
		"""Receive a sequence and use the built Tree to label subsequences"""

		labels = numpy.zeros(len(tokenizedDoc),dtype=numpy.int)

		label_value = 1 # so we can resolve adjacent entities, should they occur adjacently

		p = 0
		while p < len(tokenizedDoc):
			length = min(self.tree.getMaxKeyLength(), len(tokenizedDoc)-p)
			matchLength = self.tree.getLongestMatch(tokenizedDoc[p:p+length])
			if matchLength:
				for i in range(p,p+matchLength):
					labels[i] = label_value
				label_value += 1
				p += matchLength
			else:
				p += 1

		if return_labels:
			return labels
		elif boolean_result:
			return list(zip(tokenizedDoc, [x != 0 for x in labels]))
		else:
			return list(zip(tokenizedDoc, labels))
