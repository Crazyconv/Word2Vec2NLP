from gensim.models import Word2Vec

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

import util
import setting
from documents import Documents

import logging
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def build_average_dv(docs, doc_num, model, sentic_dic, save, save_file):
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

def build_av_tf_idf_dv(docs, doc_num, model, sentic_dic, save, save_file):
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

def build_nlp(docs, doc_num, model, sentic_dic, save, save_file):

def build_doc_vector(documents, model, build_option, sentic_dic,\
    save=True, save_file="doc_vector.bin", cluster_factor=20, num_cpus=-2):
    doc_num = documents.doc_num
    if build_option == 0:        # average
        doc_vector = build_average_dv(documents, doc_num, model, sentic_dic, save, save_file)
    elif build_option == 1:        # cluster
        doc_vector = build_av_tf_idf_dv(documents, doc_num, model, sentic_dic, save, save_file)
    elif build_option == 2:        # average + senticNet
        doc_vector = build_av_sn(documents, doc_num, model, sentic_dic, save, save_file)

    if(setting.to_normalize):
        doc_vector = normalize(doc_vector, copy=False)
    if(setting.to_scale):
        doc_vector = scale(doc_vector, copy=False)

    return doc_vector

if __name__ == "__main__":
    model = Word2Vec.load(setting.model_name)
    with open(setting.sentic_corpus, 'r') as f:
        sentic_dic = json.load(f)

    for i in range(2,3):
        logger.debug("use %s to build doc vector", setting.build_methods[i])
        for j in range(3):
            # train
            documents = Documents(setting.dbprefix + `j` + "/train")
            fv = build_doc_vector(documents, model, i, sentic_dic, True, setting.saveprefix + "train_fv_" + `i` + "_" + `j`)
            print fv.shape
            label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
            np.save(setting.saveprefix + "train_label_" + `i` + "_" + `j`, label)
            # test
            documents = Documents(setting.dbprefix + `j` + "/test")
            fv = build_doc_vector(documents, model, i, sentic_dic, True, setting.saveprefix + "test_fv_" + `i` + "_" + `j`)
            print fv.shape
            label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
            np.save(setting.saveprefix + "test_label_" + `i` + "_" + `j`, label)

