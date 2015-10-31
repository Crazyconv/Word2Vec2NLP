from gensim.models import Word2Vec
from sentences import Sentences
from util import *
import wordvector

import logging
import timeit
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

# keywords = ['apple', '']

dataset = "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/all"
model_name = "model.bin"
pre_model_name = "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/GoogleNews-vectors-negative300.bin.gz"

w2v_option = Word2VecOption(num_features=500, min_word_count=10, \
    num_workers=4, context=10, downsampling=1e-7)
csv_option = CsvOption(deli=",", title=["review", "sentiment"], \
    chunksize=100, review_name="review", sentiment_name="sentiment")
process_option = ProcessOption(rm_html=True, rm_punc=True, rm_num=True, \
    lower_case=True, rm_stop_words=False)

train_sentences = Sentences(dataset, csv_option, process_option)

# build model
if(os.path.isfile(model_name)):
    model = Word2Vec.load(model_name)
    logger.debug("model %s already exist, stop training wordvector", model_name)
else:
    logger.info("start trainning word vector")
    start_time = timeit.default_timer()
    model = wordvector.build_word_vector(train_sentences, w2v_option, save=True, save_file=model_name)
    logger.info("model %s trained in %.4lfs", model_name, timeit.default_timer() - start_time)

# find most similar words:
for word in keywords:
    print word
    print model.most_similar(word, topn=10);

# load pre-trained google news model
logger.info("start loading pre-trained dataset")
start_time = timeit.default_timer()
pre_model = Word2Vec.load_word2vec_format(pre_model_name, binary=True)
logger.info("pre-trained dataset loaded in %.4lfs", timeit.default_timer() - start_time)

# find most similar words:
for word in keywords:
    print word
    print pre_model.most_similar(word, topn=10);