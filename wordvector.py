from gensim.models import Word2Vec
from util import ProcessOption, Word2VecOption
import util

def extract_sentences(dir_name, process_option):
    for file_name in os.listdir(dir_name):
        with open(file_name, 'r') as data:
            for document in data:
                for sentence in util.document2sentences(document, process_option):
                    yield sentence


def build_word_vector(dir_name, process_option=ProcessOption() w2v_option=Word2VecOption(), save=True, save_file="model.bin"):
    num_features = w2v_option.num_features
    min_word_count = w2v_option.min_word_count
    num_workers = w2v_option.num_workers
    context = w2v_option.context
    downsampling = w2v_option.downsampling


    model = Word2Vec(extract_sentences(dir_name, process_option), workers=num_workers, size=num_features, \
        min_count=min_word_count, window=context, sample=downsampling, seed=1)
    model.init_sims(replace=True)

    if save:
        model.save(save_file)

    return model