import pandas
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from util import CsvOption, ProcessOption
import util
import os
import csv

class Sentences(object):
    def __init__(self, dir_name, csv_option=CsvOption(), process_option=ProcessOption()):
        self.dir_name = dir_name
        self.csv_option = csv_option
        self.process_option = process_option
        self.doc_num = self.get_num_doc()

    def __iter__(self):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        tknzr = TweetTokenizer(preserve_case=False);
        stop_words = set(stopwords.words("english"))
        for paragraph in self.paragraph_iterator():
            raw_sentences = tokenizer.tokenize(paragraph.decode('utf8').strip())
            for raw_sentence in raw_sentences:
                # process the sentence
                sentence = util.process(raw_sentence, tknzr, self.process_option, stop_words)
                yield sentence

    def sentiment_iterator(self):
        for file_name in os.listdir(self.dir_name):
            if file_name.endswith(".csv"):
                paragraphs = pandas.read_csv(os.path.join(self.dir_name, file_name), \
                    delimiter=self.csv_option.deli, names=self.csv_option.title, \
                    iterator=True, chunksize=self.csv_option.chunksize)
                for chunk in paragraphs:
                    for sentiment in chunk[self.csv_option.sentiment_name]:
                        yield sentiment

    def paragraph_iterator(self):
        for file_name in os.listdir(self.dir_name):
            if file_name.endswith(".csv"):
                paragraphs = pandas.read_csv(os.path.join(self.dir_name, file_name), \
                    delimiter=self.csv_option.deli, names=self.csv_option.title, \
                    iterator=True, chunksize=self.csv_option.chunksize)
                for chunk in paragraphs:
                    for paragraph in chunk[self.csv_option.review_name]:
                        yield paragraph

    def get_num_doc(self):
        num = 0
        for file_name in os.listdir(self.dir_name):
            if file_name.endswith(".csv"):
                num += self.row_count(os.path.join(self.dir_name, file_name))
        return num

    def row_count(self, file_name):
        count = 0
        with open(file_name) as csv_file:
            count += sum(1 for row in csv.reader(csv_file))
        return count