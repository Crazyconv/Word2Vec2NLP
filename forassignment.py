from gensim.models import Word2Vec
import wordvector

import logging
import timeit
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

# keywords = ['apple', '']

def compare(dataset, model_name, pre_model_name):

    # build model
    if(os.path.isfile(model_name)):
        model = Word2Vec.load(model_name)
        logger.debug("model %s already exist, stop training wordvector", model_name)
    else:
        logger.info("start trainning word vector")
        start_time = timeit.default_timer()
        model = wordvector.build_word_vector(dataset, save=True, save_file=model_name)
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

if __name__ == "__main__":
    dataset = "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/all"
    model_name = "model.bin"
    pre_model_name = "/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/GoogleNews-vectors-negative300.bin.gz"
    compare(dataset, model_name, pre_model_name)