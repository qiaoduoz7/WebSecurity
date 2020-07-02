import tensorflow as tf
from tensorflow import keras
imdb = keras.datasets.imdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.layers import Embedding, LSTM, GRU, Dropout, Dense, Input
from keras.models import Model, Sequential, load_model
from keras.preprocessing import sequence
from keras.datasets import imdb
from gensim.models.word2vec import Word2Vec

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)
# 数据连接
X_all = (list(train_x) + list(test_x))[0: 1000]
y_all = (list(train_y) + list(test_y))[0: 1000]
# 字典{word: word_index}
imdb_word2idx = imdb.get_word_index()
imdb_idx2word = dict((idx, word) for (word, idx) in imdb_word2idx.items())
imdb_idx2word['<PAD>'] = 0
imdb_idx2word['<START>'] = 1
imdb_idx2word['<UNK>'] = 2
imdb_idx2word['<UNUSED>'] = 3
X_all = [[imdb_idx2word.get(idx-3, '?') for idx in sen][1:] for sen in X_all]
print(X_all[0])
# def get_words(sent_ids):
#     return ' '.join([id2word.get(i, '?') for i in sent_ids])
# def get_words(sent_ids):
#     '''分好词
#
#     :param sent_ids:
#     :return:
#     '''
#     return [id2word.get(i, '?') for i in sent_ids]
# sent = get_words(train_x[0])


def train_word2vec(sentenceList, embedSize=300, epoch_num=10):
    '''训练word2Vec模型

    :param sentenceList:
    :param embedSize:
    :param epoch_num:
    :return:
    '''
    word2vec_model = Word2Vec(sentences=sentenceList, hs=0, negative=5, min_count=5, window=5, iter=epoch_num,
                              size=embedSize)
    # w2vModel.save(inPath + 'w2vModel/')
    return word2vec_model

def build_embedMatrix(word2vec_model):
    '''构建word2Vec模型

    :param word2vec_model:
    :return:
    '''
    word2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。
    vocab_list = [(w, word2vec_model.wv[w]) for w, v in word2vec_model.wv.vocab.items()]
    embedMatrix = np.zeros((len(word2vec_model.wv.vocab.items()) + 1, word2vec_model.vector_size))  # +1是因为停用词
    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]  # vocab_list[i][0]是词
        word2idx[word] = i + 1  # 因为加了停用词，所以其他索引都加1
        embedMatrix[i + 1] = vocab_list[i][1]  # vocab_list[i][1]这个词对应embedding_matrix的那一行
    return word2idx, embedMatrix

word2vec_model = train_word2vec(X_all, embedSize=300, epoch_num=10)
word2idx, embedMatrix = build_word2idx_embedMatrix(word2vec_model)  # 制作word2idx和embedMatrix

def make_deepLearn_data(sentenList, word2idx):
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]  # 之前都是通过word处理的，这里word2idx讲word转为id
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=MAX_SEQ_LEN))  # padding成相同长度
    return X_train_idx
X_all_idx = make_deepLearn_data(X_all, word2idx)  # 制作符合要求的深度学习数据
y_all_idx = np.array(y_all)  # 一定要注意，X_all和y_all必须是np.array()类型，否则报错

X_tra_idx, X_val_idx, y_tra_idx, y_val_idx = train_test_split(X_all_idx, y_all_idx, test_size=0.2,
                                                              random_state=0, stratify=y_all_idx)


def Lstm_model(embedMatrix):
    input_layer = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                                weights=[embedMatrix],  # 表示直接使用预训练的词向量
                                trainable=False)(input_layer)  # False表示不对词向量微调
    Lstm_layer = LSTM(units=20, return_sequences=False)(embedding_layer)
    drop_layer = Dropout(0.5)(Lstm_layer)
    dense_layer = Dense(units=1, activation="sigmoid")(drop_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


model = Lstm_model(embedMatrix)
model.fit(X_tra_idx, y_tra_idx, validation_data=(X_val_idx, y_val_idx),
          epochs=1, batch_size=100, verbose=1)
y_pred = model.predict(X_val_idx)
y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

print(f1_score(y_val_idx, y_pred_idx))
print(confusion_matrix(y_val_idx, y_pred_idx))

