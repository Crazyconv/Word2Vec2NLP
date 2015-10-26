import pandas
import nltk.data
from nltk.corpus import stopwords
from util import CsvOption, ProcessOption
import util
import os


class Sentences(object):
	def __init__(self, dir_name, csv_option=CsvOption(), process_option=ProcessOption()):
		self.dir_name = dir_name
		self.csv_option = csv_option
		self.process_option = process_option

	def sentiment_iterator(self):
		for file_name in os.listdir(self.dir_name):
			paragraphs = pandas.read_csv(os.path.join(self.dir_name, file_name), \
				delimiter=self.csv_option.deli, names=self.csv_option.title, \
				iterator=True, chunksize=self.csv_option.chunksize)
			for chunk in paragraphs:
				for sentiment in chunk[self.csv_option.sentiment_name]:
					yield sentiment

	def paragraph_iterator(self):
		for file_name in os.listdir(self.dir_name):
			paragraphs = pandas.read_csv(os.path.join(self.dir_name, file_name), \
				delimiter=self.csv_option.deli, names=self.csv_option.title, \
				iterator=True, chunksize=self.csv_option.chunksize)
			for chunk in paragraphs:
				for paragraph in chunk[self.csv_option.review_name]:
					yield paragraph

	def __iter__(self):
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		stop_words = set(stopwords.words("english"))
		for paragraph in self.paragraph_iterator():
			raw_sentences = tokenizer.tokenize(paragraph.decode('utf8').strip())
			for raw_sentence in raw_sentences:
				# process the sentence
				sentence = util.process(raw_sentence, self.process_option, stop_words)
				yield sentence
