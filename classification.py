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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def classification(option):
    for i in range(4):
        logger.debug("========== %s ==========", setting.build_methods[i])
        accuracy = 0
        precision = 0
        recall = 0
        fscore = 0
        for j in range(3):
            if(option == 0): # random forest
                classifier = RandomForestClassifier(n_estimators=100, n_jobs=100)
            elif(option == 1): # SVM
                classifier = LinearSVC()
            elif(i == 2): # naive bayes, binary
                classifier = BernoulliNB()
            else: #naive bayes, not binary
                classifier = GaussianNB()

            train_fv = np.load(setting.saveprefix + "train_fv_" + `i` + "_" + `j` + ".npy")
            train_label = np.load(setting.saveprefix + "train_label_" + `j` + ".npy")
            classifier.fit(train_fv, train_label)
            test_fv = np.load(setting.saveprefix + "test_fv_" + `i` + "_" + `j` + ".npy")
            test_label = np.load(setting.saveprefix + "test_label_" + `j` + ".npy")
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
        print "accuracy: ", accuracy/3.0
        print "precision: ", precision/3.0
        print "recall: ", recall/3.0
        print "fscore: ", fscore/3.0
        print "*****************************"

if __name__ == "__main__":
    for option in range(3):
        logger.debug("===================== %s =====================", setting.classifiers[option])
        classification(option)