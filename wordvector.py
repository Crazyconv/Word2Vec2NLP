from gensim.models import Word2Vec
from nltk.corpus import stopwords

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

from sentences import Sentences
from util import *
import util

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
    doc_vector = np.zeros((doc_num, num_features), dtype="float64")
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

def build_av_tf_idf_dv(docs, doc_num, model, num_features, save=True, save_file="doc_vector_tfidf.bin"):
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    count_fv = vectorizer.fit_transform(docs)
    tfidf_fv = tfidf_transformer.transform(count_fv)

    # {word: index}
    vocabulary = vectorizer.vocabulary_

    num_features = model.syn0.shape[1]
    doc_vector = np.zeros((doc_num, num_features), dtype="float32")
    word_set = set(model.index2word)

    index = 0
    for words in docs:
        count = 0
        for word in words:
            if word in word_set:
                doc_vector[index] = doc_vector[index] + model[word]*tfidf_fv[vocabulary[word]]
                count += 1
        doc_vector[index] = doc_vector[index] / (count+1)
        index += 1

    if save:
        np.save(save_file, doc_vector)

    return doc_vector

def build_cluster_dv(docs, doc_num, model, cluster_factor, num_cpus, save=True, save_file="doc_vector_cluster.bin"):
    word_vectors = model.syn0
    num_clusters = word_vectors[0] / cluster_factor
    clustering = KMeans(n_clusters=num_clusters, n_jobs=num_cpus)
    centroid_ids = clustering.fit_predict(word_vectors)
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


def build_doc_vector(dir_name, model, build_option, process_option=ProcessOption(), cluster_factor=5, num_cpus=-2):
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




# "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset"    