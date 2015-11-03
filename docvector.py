from gensim.models import Word2Vec

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

import util
import setting

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

def build_cluster_dv(docs, doc_num, model, cluster_factor, num_cpus, save=True, save_file="doc_vector_cluster.bin"):
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / cluster_factor
    clustering = KMeans(n_clusters=num_clusters, n_jobs=100)

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


def build_doc_vector(documents, model, build_option, \
    save=True, save_file="doc_vector.bin", cluster_factor=20, num_cpus=-2):
    doc_num = documents.doc_num
    if build_option == 1:        # average
        doc_vector = build_average_dv(documents, doc_num, model, save, save_file)
    elif build_option == 2:        # cluster
        doc_vector = build_av_tf_idf_dv(documents, doc_num, model, save, save_file)
    else:
        doc_vector = build_cluster_dv(documents, doc_num, model, cluster_factor, num_cpus, save, save_file)

    if(setting.to_normalize):
        doc_vector = normalize(doc_vector, copy=False)
    if(setting.to_scale):
        doc_vector = scale(doc_vector, copy=False)

    return doc_vector

# "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset"    