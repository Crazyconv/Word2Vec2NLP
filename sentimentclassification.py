from gensim.models import Word2Vec

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import grid_search

import os
import logging
import timeit

from sentences import Sentences
from util import *
import wordvector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def main(train_dir, test_dir):

    # these may be function parameters 
    w2v_option = Word2VecOption(num_features=300, min_word_count=40, \
        num_workers=4, context=10, downsampling=1e-3)
    csv_option = CsvOption(deli=",", title=["id", "review", "sentiment"], \
        chunksize=100, review_name="review", sentiment_name="sentiment")
    process_option = ProcessOption(rm_html=True, rm_punc=True, rm_num=True, \
        lower_case=True, rm_stop_words=False)
    model_name = "model.bin"
    save_model = True
    save_fv = True
    train_fv_name = "train_fv.bin"
    test_fv_name = "test_fv.bin"
    build_option = 1

    # logger info
    build_method = "average word vector"
    if build_option == 2:
        build_method = "average word vector with tf-idf"
    elif build_option == 3:
        build_method = "cluster word vector"
    logger.debug("text process option: %s", str(process_option))
    logger.debug("use %s to build doc vector", build_method)

    train_sentences = Sentences(train_dir, csv_option, process_option)
    logger.info("number of docs: %d", train_sentences.doc_num)

    # train word2vec
    if(os.path.isfile(model_name)):
        model = Word2Vec.load(model_name)
        logger.debug("model %s already exist, stop training wordvector", model_name)
    else:
        logger.info("start trainning word vector")
        start_time = timeit.default_timer()
        model = wordvector.build_word_vector(train_sentences, w2v_option, save=True, save_file=model_name)
        logger.info("model %s trained in %.4lfs", model_name, timeit.default_timer() - start_time)

    # get doc vector
    logger.info("start building training set doc vector")
    start_time = timeit.default_timer()
    train_fv = wordvector.build_doc_vector(train_dir, model, build_option, process_option, save_fv, train_fv_name)
    logger.info("training set doc vector built in %.4lfs", timeit.default_timer() - start_time)
    logger.info("training set doc vector saved to %s", train_fv_name)
    logger.debug("training size: %s", str(train_fv.shape))

    # train classifier
    logger.info("start training classifier")
    start_time = timeit.default_timer()
    clf = grid_search.GridSearchCV(svm.LinearSVC(), {'C':[ 0.01, 0.1, 1, 10, 100, 1000]}, cv=5, scoring = 'f1', n_jobs=10)
    best_model = clf.fit(train_fv, list(train_sentences.sentiment_iterator()))
    logger.info("finished training classifier in %.4lfs", timeit.default_timer() - start_time)

    # evaluate on test set
    logger.info("start building test set doc vector")
    start_time = timeit.default_timer()
    test_sentences = Sentences(test_dir, csv_option, process_option)
    test_fv = wordvector.build_doc_vector(test_dir, model, build_option, process_option, save_fv, test_fv_name)
    logger.info("test set doc vector built in %.4lfs", timeit.default_timer() - start_time)
    logger.info("test set doc vector saved to %s", test_fv_name)
    logger.debug("test size: %s", str(test_fv.shape))

    logger.info("start predicting test set sentiment")
    start_time = timeit.default_timer()
    predicted_sentiment = best_model.predict(test_fv)
    logger.info("finished prediction in %.4lfs", timeit.default_timer() - start_time)

    accuracy = np.mean(PredictedLabels == list(test_sentences.sentiment_iterator()))

    print "Test Set Accuracy = ", Accuracy 
    print metrics.classification_report(list(test_sentences.sentiment_iterator()), \
        predicted_sentiment, target_names=['positive', 'negative'])

if __name__ == "__main__":
    main("/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/train", \
        "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/test"
        )