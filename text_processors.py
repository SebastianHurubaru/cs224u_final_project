

import re
import os
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re


class TextProcessor(object):

	def __init__(self):
		pass

	def process_text(self, text: tf.Tensor) -> tf.Tensor:
		"""
		Takes a block of text (eg. the Item 7 of a 10-K report) and
		does pre-processing on the text.  Pre-processing steps include
		things like removing stop words, removing all non-alphabetic
		characters, lowercasing all text, etc.

		"""
		processed_text = self._process_text(text)

		# TODO - additional text processing
		return processed_text

	def _process_text(self, text: str) -> str:
		stop_words = set(stopwords.words('english'))

		# Only keep alphabetic letters in str (discard integers, punctuation, etc.)
		alpha_only = re.sub('[^a-zA-Z]+', ' ', text)
		tokens = word_tokenize(alpha_only)

		# Remove stop words
		processed_text = [token for token in tokens if not token in stop_words]

		# Lowercase all of the text and return it as one large str (same structure as input `text`) 
		return " ".join(processed_text).lower()


	def get_token_counts(self, text: str) -> dict:
		"""
		Helper function for getting the individual token 
		counts (word frequency) from a str of text.

		"""
		token_counts = dict()
		tokens = text.split()

		for token in tokens:
			if token in token_counts:
				token_counts[token] += 1
			else:
				token_counts[token] = 1

		return sorted(token_counts.items(), key=lambda x:x[1], reverse=True)
