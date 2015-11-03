from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk.data

import re

sentence_tknzr = nltk.data.load('tokenizers/punkt/english.pickle')
tweet_tknzr = TweetTokenizer(preserve_case=False)
stop_words = set(stopwords.words("english"))

def process(sentence, process_option):
    # remove html markup
    if process_option.rm_html:
        sentence = BeautifulSoup(sentence, "html.parser").get_text()

    # # remove punctuation and numbers
    # if process_option.rm_punc and process_option.rm_num:
    #     sentence = re.sub("[^a-zA-Z]", " ", sentence)
    # elif process_option.rm_punc:
    #     sentence = re.sub("[^a-zA-Z0-9]", " ", sentence)
    # elif process_option.rm_num:
    #     sentence = re.sub("[0-9]", " ", sentence)
    
    # # lower case and tokenization
    # if process_option.lower_case:
    #     sentence = sentence.lower()

    # words = sentence.split()

    #################################################################
    words = tweet_tknzr.tokenize(sentence);
    punctuation = set([';', ':', ',', '.', '!', '?', '\'', '"', \
    "$", "@", "#", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}"\
    "-", "+", "_", "=", "<", ">"])
    words = [word for word in words if not word in punctuation]


    # remove stop words
    if process_option.rm_stop_words:
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

def document2sentences(document, process_option):
    raw_sentences = tokenizer.tokenize(document.decode('utf8').strip())
    for raw_sentence in raw_sentences:
        sentence = process(raw_sentence, tweet_tknzr, process_option, stop_words)
        yield sentence