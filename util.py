from bs4 import BeautifulSoup
import re

class CsvOption(object):
	def __init__(self, deli=",", title=["id", "review"], chunksize=100):
		self.deli = deli;
		self.title = title;
		self.chunksize = chunksize;

class ProcessOption(object):
	def __init__(self, rm_html=True, rm_punc=True, rm_num=True, lower_case=True, rm_stop_words=False):
		self.rm_html = rm_html;
		self.rm_punc = rm_punc;
		self.rm_num = rm_num;
		self.lower_case = lower_case;
		self.rm_stop_words = rm_stop_words;

def process(sentence, process_option, stop_words):
	# remove html markup
	if process_option.rm_html:
		sentence = BeautifulSoup(sentence).get_text()

	# remove punctuation and numbers
	if process_option.rm_punc and process_option.rm_num:
		sentence = re.sub("[^a-zA-Z]", " ", sentence);
	elif process_option.rm_punc:
		sentence = re.sub("[^a-zA-Z0-9]", " ", sentence);
	elif process_option.rm_num:
		sentence = re.sub("[0-9]", " ", sentence);
	
	# lower case and tokenization
	if process_option.lower_case:
		sentence = sentence.lower();

	words = sentence.split();

	# remove stop words
	if process_option.rm_stop_words:
		words = [word for word in words if not word in stop_words]

	return words
