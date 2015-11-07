# Word2Vec2NLP
Using Word2Vec for sentiment analysis

## requirements

* Python 2.7
* Check requirements in *requirements.txt*
```
pip install -r requirements.txt
```
* If installation fails, or errors occur when running the application, following [this](https://virtualenv.readthedocs.org/en/latest/) to install virtualenv and install the requirements in the virtual environment. **Recommended**

## Usage

* Directory structure and data preparation
  * create a directory called `save` in root
  * if you want to run `similarity.py`, download the google news corpus from [this](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz), and put it to directory `dataset`
* Run the following application scripts, based on your interest.

## Application scripts

* **wordvector.py**: uses Word2Vec to train and compute word vectors for our corpus 
  * The corpus file is in *dataset/all*
  * The model is saved in *save/model.bin*
* **docvector.py**: extracts feature vectors using four algorithms, and save them to directory *save*
  * there four algorithms are:
    * Word Averaging
    * Word Averaging + SentiWordNet
    * NLPF (Feature sets extracted in assignment 5.4)
    * NLPF + SentiWordNet
    * For each algorithms, six feature vectors are produced, as we are to perform 3-fold cross validation, and we have a feature set and a test set for each
* **classification.py**: apply three machine learning classifiers to the feature vectors produced by `docvector.py` and print performance metrics for each combination of classifier and feature vector
  * Random Forest
  * SVM
  * Naive Bayes
* **similarity.py**: uses the word vectors computed by `wordvector.py`, and predefined vectors trained from Google News Dataset to find the 10 most similar words for the following keywords:
  * apple
  * iphone
  * ipad
  * ipod
  * mac
  * macbook 

## Good Luck!

