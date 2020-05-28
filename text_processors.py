

import re
import os
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
import re

from transformers import BertTokenizer


class TextProcessor(object):

	def __init__(self, args):
		super().__init__()

		self.args = args

	def process_text(self, text: tf.Tensor) -> tf.Tensor:
		"""
		Takes a block of text (eg. the Item 7 of a 10-K report) and
		does pre-processing on the text.  Pre-processing steps include
		things like removing stop words, removing all non-alphabetic
		characters, lowercasing all text, etc.

		"""
		# Decode the text from the Tensor
		decoded_text_list = tf.gather(text, 0).numpy().decode('utf-8', 'ignore')
		decoded_text = " ".join(decoded_text_list.split())

		processed_text = self._process_text(decoded_text)

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

class SentenceBasedTextProcessor(TextProcessor):

	def __init__(self, args):

		super().__init__(args)

		# Load pre-trained model tokenizer (vocabulary)
		bert_version = 'bert-large-cased-whole-word-masking'
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_version)

	def process_text(self, text: tf.Tensor) -> tf.Tensor:

		output = []

		# Process each text separately
		for i in range(self.args.number_of_periods):

			input_text = text[i].numpy().decode('utf-8', 'ignore')

			# Use nltk to split in sentences
			sentences = sent_tokenize(input_text)

			# Tokenize each sentence
			sentences_tokens = [self.bert_tokenizer.encode(sentence.replace('\n', ' '), add_special_tokens=True) for sentence in sentences]

			# Pad 0s to make all sentences have the same size
			sentences_tokens = [sentence + [0] * (self.args.max_sentence_length - len(sentence))
								for sentence in sentences_tokens]

			# Pad with empty lists to make all documents have the same size
			sentences_tokens += [[0] * self.args.max_sentence_length] * (self.args.max_document_size - len(sentences_tokens))

			# Truncate the too big documents
			sentences_tokens = sentences_tokens[:self.args.max_document_size]

			# Truncate the too big sentences
			sentences_tokens = [sentence[:self.args.max_sentence_length] for sentence in sentences_tokens]

			if len(sentences_tokens) != self.args.max_document_size:
				raise Exception(f'Document has not exactly {self.args.max_document_size} sentences')

			for sentence in sentences_tokens:
				if len(sentence) > self.args.max_sentence_length:
					raise Exception(f'Sentence has not exactly {self.args.max_sentence_length} tokens')

			output.append(sentences_tokens)

		return output



