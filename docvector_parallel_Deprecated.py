from gensim.models import Word2Vec
from nltk.corpus import stopwords

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

from sentences import Sentences
from util import *
import util
import mputil

import timeit
import logging
from multiprocessing import Manager
import multiprocessing as mp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def initprocess(dv):
    mputil.dv = dv

def single_av_tf_idf_dv(words, word_set, index, doc_num, vocabulary, tfidf_fv):
    doc_vector = np.frombuffer(mputil.dv).reshape((doc_num, -1))
    tfidf_fv = tfidf_fv.toarray()
    count = 0
    for word in words:
        if word in word_set and word in vocabulary:
            doc_vector[index] = doc_vector[index] + word_set[word]*tfidf_fv[0][vocabulary[word]]
            count += 1
    doc_vector[index] = doc_vector[index] / (count+1)

def single_average_dv(words, word_set, index, doc_num):
    doc_vector = np.frombuffer(mputil.dv).reshape((doc_num, -1))
    count = 0
    for word in words:
        if word in word_set:
            doc_vector[index] = doc_vector[index] + word_set[word]
            count += 1
    doc_vector[index] = doc_vector[index] / (count+1)

def build_word_vector(sentences, w2v_option=Word2VecOption(), save=True, save_file="model.bin"):
    num_features = w2v_option.num_features
    min_word_count = w2v_option.min_word_count
    num_workers = w2v_option.num_workers
    context = w2v_option.context
    downsampling = w2v_option.downsampling


    model = Word2Vec(sentences, workers=num_workers, size=num_features, \
        min_count=min_word_count, window=context, sample=downsampling, seed=1)
    model.init_sims(replace=True)

    if save:
        model.save(save_file)

    return model

def build_average_dv(docs, doc_num, model, save=True, save_file="doc_vector_ave.bin"):
    num_features = model.syn0.shape[1]

    manager = Manager()
    global_doc_vector = mp.Array('d', doc_num*num_features, lock=False)
    global_word_set = manager.dict(util.get_word_vec_dict(model))

    pool = mp.Pool(initializer=initprocess, initargs=[global_doc_vector])

    index = 0
    for words in docs:
        pool.apply_async(single_average_dv, [words, global_word_set, index, doc_num, ])
        index += 1

    pool.close()
    pool.join()

    doc_vector = np.frombuffer(global_doc_vector).reshape((doc_num, num_features))
    # print doc_vector
    if save:
        np.save(save_file, doc_vector)
    return doc_vector

def build_av_tf_idf_dv(docs, doc_num, model, save=True, save_file="doc_vector_tfidf.bin"):
    docs = list(docs)
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    count_fv = vectorizer.fit_transform(util.word2sentence(docs))
    tfidf_fv = tfidf_transformer.fit_transform(count_fv)

    num_features = model.syn0.shape[1]

    manager = Manager()
    global_word_set = manager.dict(util.get_word_vec_dict(model))
    global_vocabulary = manager.dict(vectorizer.vocabulary_);
    global_doc_vector = mp.Array('d', doc_num*num_features, lock=False)

    pool = mp.Pool(initializer=initprocess, initargs=[global_doc_vector])

    index = 0
    # test(docs[0], global_word_set, 0, doc_num, global_vocabulary, global_doc_vector, global_tfidf_fv)
    for words in docs:
        pool.apply_async(single_av_tf_idf_dv, [words, global_word_set, index, doc_num, global_vocabulary, tfidf_fv[index]])
        index += 1

    pool.close()
    pool.join()

    doc_vector = np.frombuffer(global_doc_vector).reshape((doc_num, num_features))
    if save:
        np.save(save_file, doc_vector)
    return doc_vector

def build_cluster_dv(docs, doc_num, model, cluster_factor, num_cpus, save=True, save_file="doc_vector_cluster.bin"):
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / cluster_factor
    clustering = KMeans(n_clusters=num_clusters, n_jobs=100)

    logger.info("start clustering")
    start_time = timeit.default_timer()
    centroid_ids = clustering.fit_predict(word_vectors)
    logger.info("finish clustering trained in %.4lfs", timeit.default_timer() - start_time)

    word_centroid_dic = dict(zip(model.index2word, centroid_ids))
    doc_vector = np.zeros((doc_num, num_clusters), dtype="float32")

    index = 0
    for words in docs:
        for word in words:
            if word in word_centroid_dic:
                centroid_id = word_centroid_dic[word]
                doc_vector[index][centroid_id] += 1
        index += 1

    if save:
        np.save(save_file, doc_vector)

    return doc_vector


def build_doc_vector(dir_name, model, build_option, process_option=ProcessOption(), cluster_factor=20, num_cpus=-2):
    sentences = Sentences(dir_name)
    docs = sentences.paragraph_iterator()
    doc_num = sentences.doc_num
    stop_words = set(stopwords.words("english"))
    post_docs = util.process_sentences(docs, process_option, stop_words)
    if build_option == 1:        # average
        doc_vector = build_average_dv(post_docs, doc_num, model)
    elif build_option == 2:        # cluster
        doc_vector = build_av_tf_idf_dv(post_docs, doc_num, model)
    else:
        doc_vector = build_cluster_dv(post_docs, doc_num, model, cluster_factor, num_cpus)

    return doc_vector
