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
import timeit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def build_average_dv(docs, doc_num, model, save=True, save_file="doc_vector_ave.bin"):
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

def build_av_tf_idf_dv(docs, doc_num, model, save=True, save_file="doc_vector_tfidf.bin"):
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

def build_doc_vector(documents, model, build_option, \
    save=True, save_file="doc_vector.bin", cluster_factor=20, num_cpus=-2):
    doc_num = documents.doc_num
    if build_option == 0:        # average
        doc_vector = build_average_dv(documents, doc_num, model, save, save_file)
    elif build_option == 1:        # cluster
        doc_vector = build_av_tf_idf_dv(documents, doc_num, model, save, save_file)

    if(setting.to_normalize):
        doc_vector = normalize(doc_vector, copy=False)
    if(setting.to_scale):
        doc_vector = scale(doc_vector, copy=False)

    return doc_vector

if __name__ == "__main__":
    model = Word2Vec.load(setting.model_name)

    for i in range(2):
        logger.debug("use %s to build doc vector", setting.build_methods[i])
        for j in range(3):
            # train
            documents = Documents(setting.dbprefix + `j` + "/train")
            fv = build_doc_vector(documents, model, i, True, setting.saveprefix + "train_fv_" + `i` + "_" + `j`)
            print fv.shape
            label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
            np.save(setting.saveprefix + "train_label_" + `i` + "_" + `j`, label)
            # test
            documents = Documents(setting.dbprefix + `j` + "/test")
            fv = build_doc_vector(documents, model, i, True, setting.saveprefix + "test_fv_" + `i` + "_" + `j`)
            print fv.shape
            label = np.array(list(documents.field_iterator(setting.csv_option.sentiment_name)))
            np.save(setting.saveprefix + "test_label_" + `i` + "_" + `j`, label)

