from gensim.models import Word2Vec

import util

import setting
import logging
import timeit
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('sys.stdout')

def extract_sentences(dir_name):
    for file_name in os.listdir(dir_name):
        with open(os.path.join(dir_name, file_name), 'r') as data:
            for document in data:
                for sentence in util.document2sentences(document):
                    yield sentence


def build_word_vector(dir_name, save, save_file):
    num_features = setting.w2v_option.num_features
    min_word_count = setting.w2v_option.min_word_count
    num_workers = setting.w2v_option.num_workers
    context = setting.w2v_option.context
    downsampling = setting.w2v_option.downsampling


    model = Word2Vec(extract_sentences(dir_name), workers=num_workers, size=num_features, \
        min_count=min_word_count, window=context, sample=downsampling, seed=1)
    model.init_sims(replace=True)

    if save:
        model.save(save_file)

    return model

def main(dir_name, save=True, save_file="model.bin"):
    logger.info("start trainning word vector")
    start_time = timeit.default_timer()
    model = build_word_vector(dir_name, save, save_file)
    logger.info("model %s trained in %.4lfs", save_file, timeit.default_timer() - start_time)


if __name__ == "__main__":
    with open("test.txt", 'w') as f:
        for sentence in extract_sentences("/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/all"):
            print >> f, sentence