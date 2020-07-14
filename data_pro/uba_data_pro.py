# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
from gensim.models import Doc2Vec
import multiprocessing
from sklearn.preprocessing import scale
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


cmdlines_file = "../../dataset/MasqueradeDat/User7"
labels_file = "../../dataset/MasqueradeDat/label.txt"
word2ver_bin = "uba_word2vec.bin"
max_features = 300
index = 80

def get_cmdlines():
    '''
        100个操作命令作为一个操作序列
        15000个操作命令  -》  150个操作序列
    Returns:

    '''
    x = np.loadtxt(cmdlines_file, dtype=str)   # (1, 15000)
    x = x.reshape((150, 100))    # reshape  150个操作序列 每个操作序列都有100个操作命令
    y = np.loadtxt(labels_file, dtype=int, usecols=6)
    y = y.reshape((100, 1))     # 形状
    y_train = np.zeros([50, 1], int)    #  前50个序列为0
    y = np.concatenate([y_train, y])    #  与标记文件中标记的向量相加
    y = y.reshape((150, ))

    return x, y

def get_features_by_wordbag():
    '''
        词袋
    Returns:

    '''
    x_arr, y = get_cmdlines()
    x=[]
    for i,v in enumerate(x_arr):
        v = " ".join(v)
        x.append(v)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x = vectorizer.fit_transform(x)
    x_train = x[0:index, ]
    x_test = x[index:, ]
    y_train = y[0:index, ]
    y_test = y[index:,]
    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test


def get_features_by_ngram():
    '''
        n-gram
    Returns:

    '''
    x_arr, y = get_cmdlines()
    x = []
    for i,v in enumerate(x_arr):
        v = " ".join(v)
        x.append(v)
    vectorizer = CountVectorizer(
                                 ngram_range=(2, 4),
                                 token_pattern=r'\b\w+\b',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)

    x_train = x[0:index,]
    x_test = x[index:,]
    y_train = y[0:index,]
    y_test = y[index:,]
    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(x)
    x_test = transformer.transform(x_test)
    x_train = transformer.transform(x_train)

    return x_train, x_test, y_train, y_test

def  get_features_by_wordseq():
    '''
        词序列
    Returns:

    '''
    x_arr, y = get_cmdlines()
    x = []
    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(x)
    print(tokenizer.word_counts)  # OrderedDict([('i', 2), ('love', 1), ('that', 1), ('girl', 1), ('hate', 1), ('u', 1)])   词频
    print(tokenizer.word_index)  # {'i': 1, 'love': 2, 'that': 3, 'girl': 4, 'hate': 5, 'u': 6}  顺序字典
    print(tokenizer.word_docs)  # {'i': 2, 'love': 1, 'that': 1, 'girl': 1, 'u': 1, 'hate': 1})  文档词频
    print(tokenizer.index_docs)  # {1: 2, 2: 1, 3: 1, 4: 1, 6: 1, 5: 1}     文档索引

    tokennized_texts = tokenizer.texts_to_sequences(x)
    print(tokennized_texts)  # [[1, 2, 3, 4], [1, 5, 6]] 每个词由其index表示

    X_t = pad_sequences(tokennized_texts, maxlen=None)  # 转换为2d array 即矩阵形式. 每个文本的词的个数均为maxlen. 不存在的词用0表示.
    print(X_t)  # [[1 2 3 4][0 1 5 6]]
    X_t = np.array(X_t)
    # vp = tflearn.data_utils.VocabularyProcessor(max_document_length=max_features,
    #                                           min_frequency=0,
    #                                           vocabulary=None,
    #                                           tokenizer_fn=None)
    # x=vp.fit_transform(x, unused_y=None)
    # x = np.array(list(x))

    x_train = X_t[0:index, ]
    x_test = X_t[index:, ]
    y_train = y[0:index, ]
    y_test = y[index:, ]

    return x_train, x_test, y_train, y_test

def  get_features_by_wordseq_hmm():
    '''
        getfeature for hmm
    Returns:

    '''
    x_arr,y = get_cmdlines()
    x = []
    for i,v in enumerate(x_arr):
        v=" ".join(v)
        x.append(v)

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(x)
    # print(tokenizer.word_counts)  # OrderedDict([('i', 2), ('love', 1), ('that', 1), ('girl', 1), ('hate', 1), ('u', 1)])   词频
    # print(tokenizer.word_index)  # {'i': 1, 'love': 2, 'that': 3, 'girl': 4, 'hate': 5, 'u': 6}  顺序字典
    # print(tokenizer.word_docs)  # {'i': 2, 'love': 1, 'that': 1, 'girl': 1, 'u': 1, 'hate': 1})  文档词频
    # print(tokenizer.index_docs)  # {1: 2, 2: 1, 3: 1, 4: 1, 6: 1, 5: 1}     文档索引

    tokennized_texts = tokenizer.texts_to_sequences(x)

    X_t = pad_sequences(tokennized_texts, maxlen=max_features)  # 转换为2d array 即矩阵形式. 每个文本的词的个数均为maxlen. 不存在的词用0表示.

    X_t = np.array(X_t)


    x_train = X_t[0:50, ]
    x_test = X_t[50:, ]
    y_train = y[0:50, ]
    y_test = y[50:, ]

    return x_train, x_test, y_train, y_test


def buildWordVector(imdb_w2v, text, size):
    '''
        build vector by word2vec
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


def  get_features_by_word2vec():
    '''
        build word2vec model
    Returns:

    '''
    x_all = []
    x_arr, y = get_cmdlines()
    x=[]
    for i,v in enumerate(x_arr):
        v = " ".join(v)
        x.append(v)

    for i in range(1,30):
        filename = "../../dataset/MasqueradeDat/User%d" % i
        with open(filename) as f:
            x_all.append([w.strip('\n') for w in f.readlines()])

    cores = multiprocessing.cpu_count()
    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model = gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=5, min_count=1, iter=20, workers=cores)
        model.build_vocab(x_all)
        model.train(x_all, total_examples=model.corpus_count, epochs=20)
        #model.save(word2ver_bin)

    x = np.concatenate([buildWordVector(model, z, max_features) for z in x])
    x = scale(x)


    x_train = x[0:index,]
    x_test = x[index:,]

    y_train = y[0:index,]
    y_test = y[index:,]

    return x_train, x_test, y_train, y_test