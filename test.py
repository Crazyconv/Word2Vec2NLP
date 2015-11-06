import setting
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import logging
import setting
import json
import docvector


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

with open(setting.sentic_corpus, 'r') as f:
    sentic_dic = json.load(f)

train_fv, train_label, test_fv, test_label = docvector.build_nlp_sn('./dataset/train.p', './dataset/test.p', sentic_dic)

for option in range(3):
    logger.debug("===================== %s =====================", setting.classifiers[option])

    accuracy = 0
    precision = 0
    recall = 0
    fscore = 0
    if(option == 0): # random forest
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=100)
    elif(option == 1): # SVM
        classifier = LinearSVC()
    else: 
        classifier = BernoulliNB()


    classifier.fit(train_fv, train_label)
    predicted_sentiment = classifier.predict(test_fv)

    single_accuracy = np.mean(predicted_sentiment == test_label)
    report = metrics.classification_report(test_label, \
            predicted_sentiment, target_names=['0', '1'])
    reports = report.split()[-4: -1]

    accuracy += single_accuracy
    precision += float(reports[0])
    recall += float(reports[1])
    fscore += float(reports[2])
    print "accuracy: ", single_accuracy
    print report

    print "********** average **********"
    print "accuracy: ", accuracy
    print "precision: ", precision
    print "recall: ", recall
    print "fscore: ", fscore
    print "*****************************"


