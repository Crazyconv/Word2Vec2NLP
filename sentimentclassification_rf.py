from gensim.models import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import grid_search
from sklearn.externals import joblib
import numpy as np

import os
import logging
import timeit

import setting
import docvector
from documents import Documents

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def main(model_name, train_dir, test_dir):

    # logger info
    build_method = "average word vector"
    if setting.build_option == 2:
        build_method = "average word vector with tf-idf"
    elif setting.build_option == 3:
        build_method = "cluster word vector"
    logger.debug("text process option: %s", str(setting.process_option))
    logger.debug("use %s to build doc vector", build_method)

    model = Word2Vec.load(model_name)
    logger.info("finish loading model %s", model_name)

    # get doc vector
    logger.info("start building training set doc vector")
    start_time = timeit.default_timer()
    train_documents = Documents(train_dir)
    train_fv = docvector.build_doc_vector(train_documents, model, setting.build_option, setting.save_fv, setting.train_fv_name)
    print train_fv
    logger.info("training set doc vector built in %.4lfs", timeit.default_timer() - start_time)
    logger.info("training set doc vector saved to %s", setting.train_fv_name)
    logger.debug("training size: %s", str(train_fv.shape))

    # train classifier
    logger.info("start training classifier")
    start_time = timeit.default_timer()
    forest = grid_search.GridSearchCV(RandomForestClassifier(), {'n_estimators':[100], 'n_jobs':[100]}, cv=5, scoring = 'f1_weighted', n_jobs=100)
    best_model = forest.fit(train_fv, list(train_documents.field_iterator(setting.csv_option.sentiment_name)))
    logger.info("finished training classifier in %.4lfs", timeit.default_timer() - start_time)


    # evaluate on test set
    logger.info("start building test set doc vector")
    start_time = timeit.default_timer()
    test_documents = Documents(test_dir)
    test_fv = docvector.build_doc_vector(test_documents, model, setting.build_option, setting.save_fv, setting.test_fv_name)
    print test_fv
    logger.info("test set doc vector built in %.4lfs", timeit.default_timer() - start_time)
    logger.info("test set doc vector saved to %s", setting.test_fv_name)
    logger.debug("test size: %s", str(test_fv.shape))

    logger.info("start predicting test set sentiment")
    start_time = timeit.default_timer()
    predicted_sentiment = best_model.predict(test_fv)
    logger.info("finished prediction in %.4lfs", timeit.default_timer() - start_time)

    accuracy = np.mean(predicted_sentiment == list(test_documents.field_iterator(setting.csv_option.sentiment_name)))

    print "Test Set Accuracy = ", accuracy 
    print metrics.classification_report(list(test_documents.field_iterator(setting.csv_option.sentiment_name)), \
        predicted_sentiment, target_names=['0', '1'])

if __name__ == "__main__":
    main("./save/model.bin", \
        "./dataset/train", \
        "./dataset/test"
        )