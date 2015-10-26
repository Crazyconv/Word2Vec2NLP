from gensim.models import Word2Vec
from sentences import Sentences

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

sentences = Sentences("/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset");

model = Word2Vec(sentences, workers=num_workers, size=num_features, \
	min_count=min_word_count, window=context, sample=downsampling, seed=1)


model_name = "trytweet.bin"
model.save(model_name)