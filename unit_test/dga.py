import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import re
from sklearn.metrics import classification_report
import xgboost as xgb
from tensorflow.keras.preprocessing import sequence
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import optimizers


# dataload
dga_file="../../dataset/dga/dga.txt"
alexa_file="../../dataset/dga/top-1m.csv"
filepath = "../../dataset/models/weights.{epoch:02d}-{val_loss:.2f}.h5"
# ModelCheckpoint('model_check/'+'ep{epoch:d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5',monitor='val_loss')

def load_alexa():
    ''' 获取alexa的排名数据 域名

    Returns:

    '''
    x=[]
    data = pd.read_csv(alexa_file, sep=",", header=None)
    x = [i[1] for i in data.values]
    return x

def load_dga():
    '''  黑样本 DGA数据

    Returns:

    '''
    x = []
    data = pd.read_csv(dga_file, sep="\t", header=None,
                      skiprows=18)
    x = [i[1] for i in data.values]
    return x

def get_feature_charseq():
    '''  字符序列模型  char->ASC->index

    Returns:

    '''
    alexa = load_alexa()
    dga = load_dga()
    x = alexa+dga
    max_features = 10000
    y = [0]*len(alexa) + [1]*len(dga)
    t = []
    for i in x:
        v = []
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)
    x = t
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train, x_test, y_train, y_test

def get_aeiou(domain):
    ''' 计算元音字母比例  ’好读’

    Args:
        domain:

    Returns:

    '''
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    '''  唯一字符个数 统计特征

    Args:
        domain:

    Returns:

    '''
    count = len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    '''  统计个数

    Args:
        domain:

    Returns:

    '''
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_feature():
    '''  特征整合

    Returns:  统计特征

    '''
    from sklearn import preprocessing
    alexa = load_alexa()
    dga = load_dga()
    v = alexa + dga
    y = [0]*len(alexa) + [1]*len(dga)
    x = []

    for vv in v:
        vvv = [get_aeiou(vv), get_uniq_char_num(vv), get_uniq_num_num(vv), len(vv)]
        x.append(vvv)
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train, x_test, y_train, y_test

def get_feature_2gram():
    '''  采用n-gram模型获取的特征

    Returns:

    '''
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0]*len(alexa) + [1]*len(dga)
    CV = CountVectorizer(
                                    ngram_range=(2, 2),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test

def get_feature_234gram():
    '''  采用n-gram模型获取的特征

    Returns:

    '''
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0]*len(alexa) + [1]*len(dga)
    CV = CountVectorizer(
                                    ngram_range=(2, 4),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test

def do_nb(x_train, x_test, y_train, y_test):
    '''
        nb算法
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_xgboost(x_train, x_test, y_train, y_test):
    '''

    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_mlp(x_train, x_test, y_train, y_test):
    '''

    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_rnn(trainX, testX, trainY, testY):
    '''

    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    max_document_length = 64
    #  make train data for model
    trainX = np.array(sequence.pad_sequences(trainX, maxlen=max_document_length, value=0.))
    testX = np.array(sequence.pad_sequences(testX, maxlen=max_document_length, value=0.))
    # Converting labels to binary vectors
    # trainY = to_categorical(trainY, num_classes=2)
    # testY = to_categorical(testY, num_classes=2)
    trainY = np.array(trainY)
    testY = np.array(testY)
    # Network building
    model = Sequential()
    # 嵌入层，每个词维度为128
    model.add(Embedding(len(testX), 64, input_length=max_document_length))
    # LSTM层，输出维度128，可以尝试着换成 GRU 试试
    model.add(LSTM(64, dropout=0.5))  # try using a GRU inst='ead, for fun
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 单神经元全连接层
    model.add(Activation('sigmoid'))  # sigmoid 激活函数层
    model.summary()  # 模型概述

    callbacks_list = [
        #  Save the model after every epoch
        # ModelCheckpoint(
        #     filepath,
        #     monitor='val_loss',
        #     verbose=0,
        #     save_best_only=True,
        #     save_weights_only=False,
        #     mode='auto',
        #     period=1
        # ),
        #  当被监测的数据不再增加，则停止训练。
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False
        ),
        #  当标准评估停止提升时，降低学习速率。
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        )
        # TensorBoard(
        #     log_dir='./logs',
        #     histogram_freq=1,
        #     batch_size=32,
        #     write_graph=True,
        #     write_grads=False,
        #     write_images=True,
        #     embeddings_freq=0,
        #     embeddings_layer_names=None,
        #     embeddings_metadata=None,
        #     embeddings_data=500,
        #     update_freq='epoch'
        # )
    ]

    # try using different optimizers and different optimizer configs
    # 这里可以尝试使用不同的损失函数和优化器
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    # 训练，迭代 15 次，使用测试集做验证（真正实验时最好不要这样做）
    print('Train...')
    model.fit(trainX, trainY, batch_size=128, epochs=10, callbacks=callbacks_list, validation_data=(testX, testY))
    # 评估误差和准确率
    score, acc = model.evaluate(testX, testY, batch_size=10)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == "__main__":
    print("Hello dga")
    print("234-gram & mlp")
    x_train, x_test, y_train, y_test = get_feature_charseq()
    # do_rnn(x_train, x_test, y_train, y_test)
    # x_train, x_test, y_train, y_test = get_feature()
    # do_xgboost(x_train, x_test, y_train, y_test)
    """
    print "text feature & nb"
    x_train, x_test, y_train, y_test = get_feature()
    do_nb(x_train, x_test, y_train, y_test)

    print "text feature & xgboost"
    x_train, x_test, y_train, y_test = get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)

    print "text feature & mlp"
    x_train, x_test, y_train, y_test = get_feature()
    do_mlp(x_train, x_test, y_train, y_test)


    print "charseq & rnn"
    x_train, x_test, y_train, y_test = get_feature_charseq()
    do_rnn(x_train, x_test, y_train, y_test)


    print "2-gram & mlp"
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_mlp(x_train, x_test, y_train, y_test)


    print "2-gram & XGBoost"
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_xgboost(x_train, x_test, y_train, y_test)

    print "2-gram & nb"
    x_train, x_test, y_train, y_test=get_feature_2gram()
    do_nb(x_train, x_test, y_train, y_test)
"""
