from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import multiprocessing
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


max_features = 128  # 每个word维度
max_document_length = 1000  # sequnence_length
vocabulary = None
doc2vec_path = "doc2vec.model"
word2vec_path = "word2vec.model"
#LabeledSentence = gensim.models.doc2vec.LabeledSentence
SentimentDocument = namedtuple('SentimentDocument', 'words tags')


def load_one_file(filename):
    '''

    :param filename:
    :return:
    '''
    x = ""
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    f.close()
    return x

def load_files_from_dir(rootdir):
    '''

    :param rootdir:
    :return:
    '''
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    '''

    :return:  x_train, x_test, y_train, y_test
    '''
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    path = "../../dataset/aclImdb/train/pos/"
    print("Load %s" % path)
    x_train = load_files_from_dir(path)
    y_train = [0]*len(x_train)
    path = "../../dataset/aclImdb/train/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    y_train += [1]*len(tmp)
    x_train += tmp
    path = "../../dataset/aclImdb/test/pos/"
    print("Load %s" % path)
    x_test = load_files_from_dir(path)
    y_test = [0]*len(x_test)
    path = "../../dataset/aclImdb/test/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    y_test += [1]*len(tmp)
    x_test += tmp

    return x_train, x_test, y_train, y_test

def get_features_by_wordbag():
    '''  词袋模型

    :return:
    '''
    x_train, x_test, y_train, y_test=load_all_files()
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print(vectorizer)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print(vectorizer)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()
    return x_train, x_test, y_train, y_test

def get_features_by_wordbag_tfidf():
    '''  tf-idf 策略加权的词袋模型

    :return:
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
    print(vectorizer)
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
    print(vectorizer)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    x_train = transformer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = transformer.transform(x_test)
    x_test = x_test.toarray()
    return x_train, x_test, y_train, y_test

def clean_text(corpus):
    ''' 内容预处理

    :param corpus:
    :return:
    '''
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

def normalize_text(text):
    '''# Convert text to lower-case and strip punctuation/symbols from words

    :param text:
    :return:
    '''
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

def listConvertNumpy(y):
    y = np.array(y)
    return y

def getsample(x, y):
    '''返回样本集

    :return:  比例划分样本集
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                                  random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test

# def getVecs(model, corpus, size):
#     vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
#     return np.array(np.concatenate(vecs), dtype='float')

# def getVecsByWord2Vec(model, corpus, size):
#     '''
#
#     :param model:
#     :param corpus:
#     :param size:
#     :return: get matricx_vector
#     '''
#
#     # x = np.zeros((max_document_length, size), dtype=float, order='C')
#     x = []
#     for text in corpus:   # 循环2.5w次
#         xx = []
#         for i, vv in enumerate(text):    # 一篇文档  vv 就是每个词
#             try:
#                 xx.append(model[vv].reshape((1,size)))
#             except KeyError:
#                 continue
#         x = np.concatenate(xx)
#     x = np.array(x, dtype='float')
#     return x

# def getVecsBydec2Vec(model, corpus, size):

def build_embedMatrix(word2vec_model):
    '''构建word2Vec嵌入

    :param word2vec_model:
    :return:  word2vec enmbed  and mapping(word -> word2vec 向量)
    '''
    word2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。  + 1
    vocab_list = [(w, word2vec_model.wv[w]) for w, v in word2vec_model.wv.vocab.items()]
    embedMatrix = np.zeros((len(word2vec_model.wv.vocab.items()) + 1, word2vec_model.vector_size))  # +1是因为停用词
    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]  # vocab_list[i][0]是词
        word2idx[word] = i + 1  # 因为加了停用词，所以其他索引都加1
        embedMatrix[i + 1] = vocab_list[i][1]  # vocab_list[i][1]这个词对应embedding_matrix的那一行
    return word2idx, embedMatrix

def make_data_word2vec(sentenList, word2idx):
    ''' 建立内容与模型之间的映射

    :param sentenList:
    :param word2idx:
    :return:
    '''
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]  # 之前都是通过word处理的，这里word2idx讲word转为id
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=max_features))  # padding成相同长度
    return X_train_idx

def  get_features_by_word2vec():
    ''' 训练word2vec模型

    :return:
    '''
    x_train, x_test, y_train, y_test = load_all_files()
    x_train = clean_text(x_train)
    x_test = clean_text(x_test)
    x = x_train + x_test
    y = y_train + y_test
    cores = multiprocessing.cpu_count()
    if os.path.exists(word2vec_path):
        print("Find cache file %s" % word2vec_path)
        model = gensim.models.Word2Vec.load(word2vec_path)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=5, min_count=10, iter=10, workers=cores)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=10)
        model.save(word2vec_path)
    word2idx, embedMatrix = build_embedMatrix(model)
    x_idx = make_data_word2vec(x, word2idx)
    y = listConvertNumpy(y)
    return x_idx, y, embedMatrix

def labelizeReviews(reviews, label_type):
    '''  get标签 关联id

    :param reviews:
    :param label_type:
    :return:
    '''
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        #labelized.append(LabeledSentence(v, [label]))
        #labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized

def  get_features_by_doc2vec():
    ''' 训练dec2vec

    :return:
    '''
    x_train, x_test, y_train, y_test = load_all_files()
    x_train = clean_text(x_train)
    x_test = clean_text(x_test)
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    x = x_train + x_test
    y = y_train + y_test
    cores = multiprocessing.cpu_count()
    #models = [
        # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
        # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #]
    if os.path.exists(doc2vec_path):
        print("Find cache file %s" % doc2vec_path)
        model = Doc2Vec.load(doc2vec_path)
    else:
        model = Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores, epochs=10,
                        window=5, alpha=0.025)
        #for model in models:
        #    model.build_vocab(x)
        model.build_vocab(x)
        #models[1].reset_from(models[0])
        #for model in models:
        #    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        #models[0].train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.train(x, total_examples=model.corpus_count, epochs=12)
        model.save(doc2vec_path)

    word2idx, embedMatrix = build_embedMatrix(model)
    x_idx = getVecsByWord2Vec(x, word2idx)
    return x_idx, y_train, y_test, embedMatrix
