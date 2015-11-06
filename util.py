from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk.data

import setting

import re

sentence_tknzr = nltk.data.load('tokenizers/punkt/english.pickle')
tweet_tknzr = TweetTokenizer(preserve_case=False)
stop_words = set(stopwords.words("english"))

def process(sentence):
    # remove html markup
    if setting.process_option.rm_html:
        sentence = BeautifulSoup(sentence, "html.parser").get_text()

    words = tweet_tknzr.tokenize(sentence);
    words = [word for word in words if re.search(r'[a-zA-Z]+', word)]

    # remove stop words
    if setting.process_option.rm_stop_words:
        words = [word for word in words if not word in stop_words]

    return words

def word2sentence(word_docs):
    for words in word_docs:
        yield " ".join(words)

def get_word_vec_dict(model):
    dic = {}
    for word in model.index2word:
        dic[word] = model[word]
    return dic

def document2sentences(document):
    try:
        raw_sentences = sentence_tknzr.tokenize(document.decode('utf8').strip())
        for raw_sentence in raw_sentences:
            sentence = process(raw_sentence)
            yield sentence
    except:
        return