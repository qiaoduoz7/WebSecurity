# -*- coding:utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers



def do_mlp(x_train, x_test, y_train, y_test):
    '''
        train mlp
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

def do_xgboost(x_train, x_test, y_train, y_test):
    '''
        train xgboost
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

def do_nb(x_train, x_test, y_train, y_test):
    '''
        train nb
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

def do_cnn(trainX, testX, trainY, testY):
    '''
        train cnn
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    # trainY = to_categorical(trainY, nb_classes=2)
    # testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    input_data = Input(shape=[100, ])
    emb = Embedding(input_dim=10000, output_dim=128)(input_data)
    cnn1 = Conv1D(filters=64, kernel_size=2, strides=1, padding='valid', activation='relu')(emb)
    cnn2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(emb)
    cnn3 = Conv1D(filters=64, kernel_size=4, strides=1, padding='valid', activation='relu')(emb)

    cnn = Concatenate(axis=1)([cnn1, cnn2, cnn3])
    # cnn_expand = expand_dims(2, axis=-1)(cnn)
    maxPool1D = MaxPool1D(pool_size=2)(cnn)
    flatten = Flatten()(maxPool1D)
    dropout = Dropout(0.2)(flatten)
    output = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=[input_data], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    model.summary()
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=64, shuffle=True)

    y_predict_list = model.predict(testX)

    y_predict = []
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))

def do_rnn_wordbag(trainX, testX, trainY, testY):
    '''
        train rnn use wordbag
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    y_test=testY
    #trainX = pad_sequences(trainX, maxlen=100, value=0.)
    #testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    model = Sequential()
    model.add(Input(shape=(100, )))
    model.add(Embedding(input_dim=1000, output_dim=128))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    model.summary()
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=64, shuffle=True)

    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        #print  i[0]
        if i[0] >= 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))

    print(y_train)

    print("ture")
    print(y_test)
    print("pre")
    print(y_predict)

def do_birnn_wordbag(trainX, testX, trainY, testY):
    '''
        train birnn user wordbag
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=100, value=0.)
    #testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Network building
    model = Sequential([
        Input(shape=(100,)),
        Embedding(input_dim=10000, output_dim=128),
        Bidirectional(LSTM(128)),
        Dropout(0.5),
        Dense(2, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    model.summary()
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=64, shuffle=True)


    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        #print  i[0]
        if i[0] >= 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))

def do_hmm(trainX, testX, trainY, testY):
    '''
        train hmm
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    T = -580
    N = 2
    lengths = [1]
    X = [[0]]
    print(len(trainX))
    for i in trainX:
        z=[]
        for j in i:
            z.append([j])
        #print z
        #X.append(z)
        X = np.concatenate([X, np.array(z)])
        lengths.append(len(i))

    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, lengths)

    y_predict=[]
    for i in testX:
        z = []
        for j in i:
            z.append([j])
        y_pred=remodel.score(z)
        print(y_pred)
        if y_pred < T:
            y_predict.append(1)
        else:
            y_predict.append(0)
    y_predict=np.array(y_predict)

    print(classification_report(testY, y_predict))
    print(metrics.confusion_matrix(testY, y_predict))

    print(testY)
    print(y_predict)

def show_hmm(trainX, testX, trainY, testY):
    '''
        get res for hmm
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    a = []
    b = []
    c = []

    N = 2
    lengths = [1]
    X = [[0]]
    print(len(trainX))
    for i in trainX:
        z=[]
        for j in i:
            z.append([j])
        X=np.concatenate([X,np.array(z)])
        lengths.append(len(i))

    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    remodel.fit(X, lengths)

    for T in range(-600,-400,10):
        y_predict = []
        for i in testX:
            z = []
            for j in i:
                z.append([j])
            y_pred = remodel.score(z)
            #print y_pred
            if y_pred < T:
                y_predict.append(1)
            else:
                y_predict.append(0)
        y_predict = np.array(y_predict)
        precision = precision_score(y_test,y_predict)
        recall = recall_score(y_test,y_predict)
        a.append(T)
        b.append(precision)
        c.append(recall)
        plt.plot(a, b,'-rD',a,c, ':g^')
        #plt.plot(a, b, 'r')
        #plt.plot(a, c, 'r')
    plt.xlabel("log probability")
    plt.ylabel("metrics.recall&precision")
    #plt.ylabel("metrics.precision")
    #plt.title("metrics.precision")
    plt.title("metrics.recall&precision")
    plt.legend()
    plt.show()