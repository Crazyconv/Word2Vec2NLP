from bs4 import BeautifulSoup
import re

class CsvOption(object):
    def __init__(self, deli=",", title=["id", "review", "sentiment"], \
        chunksize=100, review_name="review", sentiment_name="sentiment"):
        self.deli = deli
        self.title = title
        self.chunksize = chunksize
        self.review_name = review_name
        self.sentiment_name = sentiment_name

class ProcessOption(object):
    def __init__(self, rm_html=True, rm_punc=True, rm_num=True, lower_case=True, rm_stop_words=False):
        self.rm_html = rm_html
        self.rm_punc = rm_punc
        self.rm_num = rm_num
        self.lower_case = lower_case
        self.rm_stop_words = rm_stop_words

    def __str__(self):
        return str({"rm_html": self.rm_html, \
            "rm_punc": self.rm_punc, \
            "rm_num": self.rm_num, \
            "lower_case": self.lower_case, \
            "rm_stop_words": self.rm_stop_words 
            })

class Word2VecOption(object):
    def __init__(self, num_features=300, min_word_count=40, \
        num_workers=4, context=10, downsampling=1e-3):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.context = context
        self.downsampling = downsampling

def process(sentence, process_option, stop_words):
    # remove html markup
    if process_option.rm_html:
        sentence = BeautifulSoup(sentence, "html.parser").get_text()

    # remove punctuation and numbers
    if process_option.rm_punc and process_option.rm_num:
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
    elif process_option.rm_punc:
        sentence = re.sub("[^a-zA-Z0-9]", " ", sentence)
    elif process_option.rm_num:
        sentence = re.sub("[0-9]", " ", sentence)
    
    # lower case and tokenization
    if process_option.lower_case:
        sentence = sentence.lower()

    words = sentence.split()

    # remove stop words
    if process_option.rm_stop_words:
        words = [word for word in words if not word in stop_words]

    return words

def process_sentences(sentences, process_option, stop_words):
    for sentence in sentences:
        yield process(sentence, process_option, stop_words)

def word2sentence(word_docs):
    for words in word_docs:
        yield " ".join(words)

def get_word_vec_dict(model):
    dic = {}
    for word in model.index2word:
        dic[word] = model[word]
    return dic