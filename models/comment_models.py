from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from data_pro import comment_data_pro
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dropout, Dense, Input
from tensorflow.keras.models import Model, Sequential, load_model

from utils import dispaly_res


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    print("rf and doc2vec")
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
    print("MLP and doc2vec")
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_rnn_word2vec(x_train, x_test, y_train, y_test, embedMatrix):
    ''' 对rnn的训练

    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    input_layer = Input(shape=(128,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                                weights=[embedMatrix],  # 表示直接使用预训练的词向量
                                trainable=False)(input_layer)  # False表示不对词向量微调
    Lstm_layer = LSTM(units=128, return_sequences=False)(embedding_layer)
    drop_layer = Dropout(0.5)(Lstm_layer)
    dense_layer = Dense(units=1, activation="sigmoid")(drop_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=10, batch_size=32, verbose=1)
    y_pred = model.predict(x_test)
    y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

    print(f1_score(y_test, y_pred_idx))
    print(confusion_matrix(y_test, y_pred_idx))


def do_cnn_doc2vec(trainX, testX, trainY, testY, embedMatrix):
    # set parameters:  设定参数
    max_features = 128  # 最大特征数（词汇表大小）
    maxlen = 128  # 序列最大长度
    # batch_size = 32  # 每批数据量大小
    # embedding_dims = 50  # 词嵌入维度
    # nb_filter = 250  # 1维卷积核个数
    # filter_length = 3  # 卷积核长度
    # hidden_dims = 250  # 隐藏层维度
    # nb_epoch = 10  # 迭代次数
    # 构建模型

    # model build
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=max_features, output_dim=len(embedMatrix), input_length=maxlen),
        keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),
        keras.layers.MaxPool1D(2, padding='valid'),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Activation('relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 先从一个高效的嵌入层开始，它将词汇的索引值映射为 embedding_dims 维度的词向量
    # input_layer = Input(shape=(128,), dtype='int32')
    # model.add(Embedding(max_features,
    #                     len(embedMatrix),    # 词汇表大小
    #                     input_length=maxlen))    # 输出维度
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    # 添加一个 1D 卷积层，它将学习 nb_filter 个 filter_length 大小的词组卷积核
    # model.add(keras.layers.Convolution1D(nb_filter=nb_filter,
    #                         filter_length=filter_length,
    #                         border_mode='valid',
    #                         activation='relu',
    #                         subsample_length=1))
    # we use max pooling:
    # 使用最大池化
    #model.add(keras.layers.GlobalMaxPooling1D())
    # We add a vanilla hidden layer:
    # 添加一个原始隐藏层
    # model.add(keras.layers.Dense(hidden_dims))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.Activation('relu'))
    # We project onto a single unit output layer, and squash it with a sigmoid:
    # 投影到一个单神经元的输出层，并且使用 sigmoid 压缩它
    # model.add(keras.layers.Dense(1))
    # model.add(keras.layers.Activation('sigmoid'))
    model.summary()  # 模型概述

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    # 训练
    history = model.fit(trainX, trainY, batch_size=32, epochs=5, validation_data=(testX, testY))
    dispaly_res.plot_graphs(history, 'accuracy')

    # # 定义损失函数，优化器，评估矩阵
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # # 训练，迭代 nb_epoch 次
    # model.fit(trainX, trainY,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           validation_data=(testX, testY))

if __name__ == "__main__":
    # 读取数据
    # x_train, x_test, y_train, y_test = load_all_files()
    # 获取特征（词袋模型）
    # x_train, x_test, y_train, y_test = get_features_by_wordbag()
    # print(x_train)
    # get_features_by_wordbag_tfidf()
    # get_features_by_word2vec()
    # get_features_by_doc2vec
    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag()
    # x_train, x_test, y_train, y_test, embedMatrix = comment_data_pro.get_features_by_word2vec()
    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag_tfidf()
    x_idx, y, embedMatrix = comment_data_pro.get_features_by_word2vec()
    x_train, x_test, y_train, y_test = comment_data_pro.getsample(x_idx, y)

    # do_rnn_word2vec(x_train, x_test, y_train, y_test, embedMatrix)
    do_cnn_doc2vec(x_train, x_test, y_train, y_test, embedMatrix)
    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag()
    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_doc2vec()
    # do_nb_wordbag(x_train, x_test, y_train, y_test)