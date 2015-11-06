from gensim.models import Word2Vec

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

import util
import setting
from documents import Documents

import logging
import json
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def build_average_dv(docs, doc_num, model, save, save_file):
    num_features = model.syn0.shape[1]
    doc_vector = np.zeros((doc_num, num_features), dtype="float32")
    word_set = set(model.index2word)

    index = 0
    for words in docs:
        count = 0
        for word in words:
            if word in word_set:
                doc_vector[index] = doc_vector[index] + model[word]
                count += 1
        doc_vector[index] = doc_vector[index] / (count+1)
        index += 1

    if save:
        np.save(save_file, doc_vector)

    return doc_vector

def build_av_tf_idf_dv(docs, doc_num, model, save, save_file):
    docs = list(docs)
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    count_fv = vectorizer.fit_transform(util.word2sentence(docs))
    tfidf_fv = tfidf_transformer.fit_transform(count_fv)

    # {word: index}
    vocabulary = vectorizer.vocabulary_

    num_features = model.syn0.shape[1]
    doc_vector = np.zeros((doc_num, num_features), dtype="float32")
    word_set = set(model.index2word)

    index = 0
    for words in docs:
        vec = tfidf_fv[index].toarray()
        count = 0
        for word in words:
            if word in word_set and word in vocabulary:
                doc_vector[index] = doc_vector[index] + model[word] * vec[0][vocabulary[word]]
                count += 1
        doc_vector[index] = doc_vector[index] / (count+1)
        index += 1

    if save:
        np.save(save_file, doc_vector)

    return doc_vector

def build_av_sn(docs, doc_num, model, sentic_dic, save, save_file):
    docs = list(docs)

    num_features = model.syn0.shape[1]
    doc_vector = np.zeros((doc_num, num_features), dtype="float32")
    word_set = set(model.index2word)

    index = 0
    for words in docs:
        count = 0
        for word in words:
            if word in word_set:
                factor = 1
                if word+"#a" in sentic_dic:
                    factor = sentic_dic[word+"#a"]
                    if(factor > 0.5 or factor < -0.5):
                        factor = 2
                    elif(factor > 0 or factor < 0):
                        factor = 1.5
                    else:
                        factor = 1
                doc_vector[index] = doc_vector[index] + model[word] * factor
                count += 1
        doc_vector[index] = doc_vector[index] / (count+1)
        index += 1

    if save:
        np.save(save_file, doc_vector)

    return doc_vector

def build_nlp(train_file_name, test_file_name, sentic_dic, save, save_train_file, save_test_file):
    v = DictVectorizer(sparse=False)

    with open(train_file_name, 'rb') as f:
        feature_set = pickle.load(f)
    words = [words for (words, label) in feature_set]
    train_fv = v.fit_transform(words)

    with open(test_file_name, 'rb') as f:
        feature_set = pickle.load(f)
    words = [words for (words, label) in feature_set]
    test_fv = v.transform(words)

    if save:
        np.save(save_train_file, train_fv)
        np.save(save_test_file, test_fv)

    return train_fv, test_fv

def build_nlp_sn(train_file_name, test_file_name, sentic_dic, save, save_train_file, save_test_file):
    v = DictVectorizer(sparse=False)

    fv_dic = []
    with open(train_file_name, 'rb') as f:
        feature_set = pickle.load(f)
    for words, _ in feature_set:
        single_dic = {}
        for word in words:
            if(words[word] and word+"#a" in sentic_dic):
                if(word in single_dic):
                    single_dic[word] += sentic_dic[word+"#a"]
                else:
                    single_dic[word] = sentic_dic[word+"#a"]
        fv_dic.append(single_dic)

    train_fv = v.fit_transform(fv_dic)

    fv_dic = []
    with open(test_file_name, 'rb') as f:
        feature_set = pickle.load(f)
    for words, _ in feature_set:
        single_dic = {}
        for word in words:
            if(words[word] and word+"#a" in sentic_dic):
                if(word in single_dic):
                    single_dic[word] += sentic_dic[word+"#a"]
                else:
                    single_dic[word] = sentic_dic[word+"#a"]
        fv_dic.append(single_dic)

    test_fv = v.transform(fv_dic)
    if save:
        np.save(save_train_file, train_fv)
        np.save(save_test_file, test_fv)

    return train_fv, test_fv


def build_doc_vector(build_option, sentic_dic, documents=None, model=None, train_file_name=None, test_file_name=None, \
    save=True, save_file="doc_vector.bin", save_file2="doc_vector.bin"):

    doc_num = 1
    if not documents is None:
        doc_num = documents.doc_num

    if build_option == 0:        # average
        doc_vector = build_average_dv(documents, doc_num, model, save, save_file)
    elif build_option == 1:        # average + senticNet
        doc_vector = build_av_sn(documents, doc_num, model, sentic_dic, save, save_file)
    elif build_option == 2:        #nlp 
        doc_vector = build_nlp(train_file_name, test_file_name, sentic_dic, save, save_file, save_file2)
    elif build_option == 3:        #nlp +senticNet
        doc_vector = build_nlp_sn(train_file_name, test_file_name, sentic_dic, save, save_file, save_file2)
    else:     # average+ tf-idf
        doc_vector = build_av_tf_idf_dv(documents, doc_num, model, save, save_file)

    if(setting.to_normalize):
        doc_vector = normalize(doc_vector, copy=False)
    if(setting.to_scale):
        doc_vector = scale(doc_vector, copy=False)

    return doc_vector

if __name__ == "__main__":
    model = Word2Vec.load(setting.model_name)
    with open(setting.sentic_corpus, 'r') as f:
        sentic_dic = json.load(f)

    for i in range(4):
        logger.debug("use %s to build doc vector", setting.build_methods[i])
        for j in range(3):
            if(i <= 1):
                # train
                documents = Documents(setting.dbprefix + `j` + "/train")
                train_fv = build_doc_vector(i, sentic_dic, documents=documents, model=model, \
                    save_file=setting.saveprefix + "train_fv_" + `i` + "_" + `j`)
                if(i == 0):
                    train_label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
                    np.save(setting.saveprefix + "train_label_" + `j`, train_label)
                # test
                documents = Documents(setting.dbprefix + `j` + "/test")
                test_fv = build_doc_vector(i, sentic_dic, documents=documents, model=model, \
                    save_file=setting.saveprefix + "test_fv_" + `i` + "_" + `j`)
                if(i == 0):
                    test_label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
                    np.save(setting.saveprefix + "test_label_" + `j`, test_label)
            else:
                train_fv, test_fv = build_doc_vector(i, sentic_dic, \
                    train_file_name=setting.fsprefix + `j` + "/train.p", test_file_name=setting.fsprefix + `j` + "/test.p", \
                    save_file=setting.saveprefix + "train_fv_" + `i` + "_" + `j`, \
                    save_file2=setting.saveprefix + "test_fv_" + `i` + "_" + `j`)

