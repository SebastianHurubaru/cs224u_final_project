

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
		text = text.numpy().decode('utf-8', 'ignore')

		text = " ".join(text.split())

		filtered_text = self.filter_text(text)
		token_counts = self.get_token_counts(filtered_text)

		print("token_counts length: {}".format(len(token_counts)))
		print("top token_counts: {}".format(token_counts[:10]))

		print()

		tensor = tf.convert_to_tensor(filtered_text, dtype=tf.string)

		f = open("example_processed_text_output.txt", "a", newline="", encoding="utf-8")
		f.write(filtered_text)
		f.close()
		return text

	def filter_text(self, text: str) -> str:
		stop_words = set(stopwords.words('english'))

		# Only keep alphabetic letters in str (discard integers, punctuation, etc.)
		alpha_only = re.sub('[^a-zA-Z]+', ' ', text)
		tokens = word_tokenize(alpha_only)

		# Remove stop words
		filtered_text = [token for token in tokens if not token in stop_words]

		# Lowercase all of the text and return it as one large str (same structure as input `text`) 
		return " ".join(filtered_text).lower()


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
