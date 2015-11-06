class CsvOption(object):
    def __init__(self, deli=",", title=["review", "sentiment"], \
        chunksize=100, review_name="review", sentiment_name="sentiment"):
        self.deli = deli
        self.title = title
        self.chunksize = chunksize
        self.review_name = review_name
        self.sentiment_name = sentiment_name

class ProcessOption(object):
    def __init__(self, rm_html=True, rm_stop_words=False):
        self.rm_html = rm_html
        self.rm_stop_words = rm_stop_words

    def __str__(self):
        return str({"rm_html": self.rm_html, \
            "rm_stop_words": self.rm_stop_words 
            })

class Word2VecOption(object):
    def __init__(self, num_features=300, min_word_count=40, \
        num_workers=4, context=10, downsampling=1e-3):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.context = context
        self.downsampling = downsampling

w2v_option = Word2VecOption(num_features=300, min_word_count=40, \
    num_workers=4, context=10, downsampling=1e-3)
csv_option = CsvOption(deli=",", title=["review", "sentiment"], \
    chunksize=100, review_name="review", sentiment_name="sentiment")
process_option = ProcessOption(rm_html=True, rm_stop_words=True)

to_scale = False
to_normalize = False

save_fv = True
train_fv_name = "train_fv.bin"
test_fv_name = "test_fv.bin"
build_option = 1