import csv, nltk, random
from nltk.stem import *
import string
import pickle

def read_file(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        return [(str(line[0]).translate(None, string.punctuation).decode("utf8"), line[1]) for line in reader if line[1] in ['0','1']]


def tokenize(dataset):
    stemmer=SnowballStemmer('english')
    return [([stemmer.stem(word) for word in nltk.word_tokenize(sent)], label) for (sent, label) in dataset]

data = [["set2.csv", "set3.csv", "set1.csv"],\
["set1.csv", "set3.csv", "set2.csv"],\
["set1.csv", "set2.csv", "set3.csv"],]

for i in range(3):
    datum = data[i]
    tweets = read_file(datum[0])
    tweets_tokens = tokenize(tweets)
    all_tokens = [token for (tweet, label) in tweets_tokens for token in tweet]
    features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x in nltk.corpus.stopwords.words('english')]
    feature_set = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in tweets_tokens]

    tweets = read_file(datum[1])
    tweets_tokens = tokenize(tweets)
    all_tokens = [token for (tweet, label) in tweets_tokens for token in tweet]
    features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x in nltk.corpus.stopwords.words('english')]
    feature_set.extend([({feature: feature in tokens for feature in features }, label) for (tokens, label) in tweets_tokens])

    pickle.dump(feature_set, open("./dataset/FeatureSet"+`i`+'/train.p', "wb" ) )

    tweets = read_file(datum[2])
    tweets_tokens = tokenize(tweets)
    all_tokens = [token for (tweet, label) in tweets_tokens for token in tweet]
    features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x in nltk.corpus.stopwords.words('english')]
    feature_set = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in tweets_tokens]

    pickle.dump(feature_set, open("./dataset/FeatureSet"+`i`+'/test.p', "wb" ))