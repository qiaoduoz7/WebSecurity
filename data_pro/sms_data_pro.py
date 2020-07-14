from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import gensim
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers


max_features=1000
max_document_length=160
vocabulary=None
doc2ver_bin="smsdoc2ver.bin"
word2ver_bin="smsword2ver.bin"
SentimentDocument = namedtuple('SentimentDocument', 'words tags')


def load_all_files():
    '''
        load sms
    Returns:

    '''
    x=[]
    y=[]
    datafile="../../dataset/sms/smsspamcollection/SMSSpamCollection.txt"
    with open(datafile) as f:
        for line in f:
            line=line.strip('\n')
            label,text=line.split('\t')
            x.append(text)
            if label == 'ham':
                y.append(0)
            else:
                y.append(1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test


def get_features_by_wordbag():
    '''
        build wordbag
    Returns:

    '''
    x_train, x_test, y_train, y_test=load_all_files()
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def get_features_by_wordbag_tfidf():
    '''
        build tfidf
    Returns:

    '''
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 binary=True)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 min_df=1 )
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train = transformer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = transformer.transform(x_test)
    x_test = x_test.toarray()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def get_features_by_ngram():
    '''
        bulid n-gram
    Returns:

    '''
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                ngram_range=(3, 3),
                                strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 token_pattern=r'\b\w+\b',
                                 binary=True)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                ngram_range=(3, 3),
                                strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 token_pattern=r'\b\w+\b',
                                 min_df=1 )
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()    # 矩阵的稠密化
    transformer = TfidfTransformer(smooth_idf=False)
    x_train = transformer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = transformer.transform(x_test)
    x_test = x_test.toarray()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

def  get_features_by_tf():
    '''
        bulid vocabulary
    Returns:

    '''
    x_train, x_test, y_train, y_test = load_all_files()
    x_all = x_train + x_test
    y_all = y_train + y_test
    tokenizer = Tokenizer(num_words=None)
    # tokenizer.fit_on_texts(x_all)
    tokennized_texts = tokenizer.texts_to_sequences(x_all)
    x_all = pad_sequences(tokennized_texts, maxlen=max_features)
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.4)

    return x_train, x_test, y_train, y_test


def cleanText(corpus):
    '''
        clean content
    Args:
        corpus:

    Returns:

    '''
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    '''
        normalize text
    Args:
        text:

    Returns:

    '''
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def labelizeReviews(reviews, label_type):
    '''

    Args:
        reviews:
        label_type:

    Returns:

    '''
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        #labelized.append(LabeledSentence(v, [label]))
        #labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    '''
        get word vector by doc2vec
    Args:
        model:
        corpus:
        size:

    Returns:

    '''
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs),dtype='float')


def buildWordVector(imdb_w2v,text, size):
    '''
        get vector by word2vec
    Args:
        imdb_w2v:
        text:
        size:

    Returns:

    '''
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def  get_features_by_doc2vec():
    '''
        build doc2vec model
    Returns:

    '''
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    x_train = labelizeReviews(x_train, 'TRAIN')   # 添加段落标签
    x_test = labelizeReviews(x_test, 'TEST')

    x = x_train + x_test
    cores = multiprocessing.cpu_count()
    #models = [
        # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
        # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #]
    if os.path.exists(doc2ver_bin):
        print("Find cache file %s" % doc2ver_bin)
        model = Doc2Vec.load(doc2ver_bin)
    else:
        model = Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores,iter=60)
        #for model in models:
        #    model.build_vocab(x)
        model.build_vocab(x)
        #models[1].reset_from(models[0])

        #for model in models:
        #    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        #models[0].train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    #x_test=getVecs(models[0],x_test,max_features)
    #x_train=getVecs(models[0],x_train,max_features)
    x_test = getVecs(model, x_test, max_features)
    x_train = getVecs(model, x_train, max_features)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

def  get_features_by_word2vec():
    '''
        build word2vec model
    Returns:

    '''
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    x = x_train + x_test
    cores = multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)

    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = scale(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = scale(x_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


