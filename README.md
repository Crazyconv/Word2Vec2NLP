# Word2Vec2NLP
Using Word2Vec for sentiment analysis

## requirements

all in python, check requirements.txt for libraries used

## How to use

* rename `dataset_used` as `dataset`
* create a directory called `save` in root
* run `python docvector.py` to generate feature vectors (training and testing) for different algorithms
  * word2vec
  * word2vec + senticNet
  * nlp
  * nlp + senticNet
* run `python classification.py` to run classification
  * random forest
  * svm
  * naive bayes
* if you want to play with the word similarity, download the google news corpus from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz, put it to directory `dataset`, and run `python similarity.py`. It will give you the top 10 most similar words to the following words: apple, ipad, ipod, iphone, mac, macbook
  * This will give the results based on our corpus and google news corpus.
  * The size of the google news corpus is quite large. You can comment out the relevant code to skip it and just get the result from the data of our corpus

## Good Luck!

