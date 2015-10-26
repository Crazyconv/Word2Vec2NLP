import pandas;
import nltk.data;
from nltk.corpus import stopwords
from util import CsvOption, ProcessOption
import util


class Sentences(object):
	def __init__(self, file_name, csv_option=CsvOption(), process_option=ProcessOption(), chunksize=4):
		self.file_name = file_name
		self.csv_option = csv_option
		self.process_option = process_option

	def __iter__(self):
		paragraphs = pandas.read_csv(self.file_name, delimiter=self.csv_option.deli, names=self.csv_option.title, iterator=True, chunksize=self.csv_option.chunksize);
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		stop_words = set(stopwords.words("english"));
		for chunk in paragraphs:
			# split review into sentences using NLTK tokenizer
			for paragraph in chunk["review"]:
				raw_sentences = tokenizer.tokenize(paragraph.decode('utf8').strip())
				for raw_sentence in raw_sentences:
					# process the sentence
					sentence = util.process(raw_sentence, self.process_option, stop_words);
					yield sentence
