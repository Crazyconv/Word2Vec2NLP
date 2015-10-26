from gensim.models import Word2Vec

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import grid_search

import os

from sentences import Sentences
from util import *
import wordvector

def main(train_dir, test_dir, log_file):
	with open(log_file, 'w') as out_file: 

		# these may be function parameters 
		w2v_option = Word2VecOption(num_features=300, min_word_count=40, \
			num_workers=4, context=10, downsampling=1e-3)
		csv_option = CsvOption(deli=",", title=["id", "review", "sentiment"], \
			chunksize=100, review_name="review", sentiment_name="sentiment"))
		process_option = ProcessOption(rm_html=True, rm_punc=True, rm_num=True, \
			lower_case=True, rm_stop_words=False)
		model_name = "model.bin"
		save_model = True
		save_fv = True
		train_fv_name = "train_fv.bin"
		test_fv_name = "test_fv.bin"
		build_option = 1

		train_sentences = Sentences(train_file, csv_option, process_option)


		# train word2vec
		if(os.path.isfile(model_name)):
			model = Word2Vec.load(model_name)
		else:
			model = wordvector.build_word_vector(train_sentences, w2v_option, save=True, save_file=model_name)

		# get doc vector
		train_fv = wordvector.build_doc_vector(train_dir, model, build_option, process_option, save_fv, train_fv_name)

		# train classifier
		clf = grid_search.GridSearchCV(svm.LinearSVC(), {'C':[ 0.01, 0.1, 1, 10, 100, 1000]}, cv=5, scoring = 'f1', n_jobs=10)
		best_model = clf.fit(train_fv, train_sentences.sentiment_iterator()) # if cannot use iterator, use list()

		# evaluate on test set
		test_sentences = Sentences(test_file, csv_option, process_option)
		test_fv = wordvector.build_doc_vector(train_dir, model, build_option, process_option, save_fv, train_fv_name)
		predicted_sentiment = best_model.predict(test_fv)
		accuracy = np.mean(PredictedLabels == list(train_sentences.sentiment_iterator()))

		print >> log_file, "Test Set Accuracy = ", Accuracy 
		print >> log_file, metrics.classification_report(list(train_sentences.sentiment_iterator()), \
			predicted_sentiment, target_names=['positive', 'negative'])
