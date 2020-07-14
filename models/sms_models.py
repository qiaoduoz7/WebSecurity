
import os
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
from sklearn import svm
import gensim
from gensim.models import Doc2Vec
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    '''
        model = nb
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    '''
        nb and doc2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("NB and doc2vec")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_nb_word2vec(x_train, x_test, y_train, y_test):
    '''
        nb and word2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("NB and word2vec")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def do_svm_wordbag(x_train, x_test, y_train, y_test):
    '''
        svm and  wordbag
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_svm_doc2vec(x_train, x_test, y_train, y_test):
    '''
        svm and deo2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("SVM and doc2vec")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_svm_word2vec(x_train, x_test, y_train, y_test):
    '''
        svm and word2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("SVM and word2vec")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    '''

    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("rf and doc2vec")
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_rf_word2vec(x_train, x_test, y_train, y_test):
    '''
        rf and word2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("rf and word2vec")
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_cnn_wordbag(trainX, testX, trainY, testY, max_document_length):
    '''
        cnn and wordbag
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    input = Input(shape=[max_document_length, ])
    embedding = Embedding(input_dim=1000000, output_dim=128)(input)
    cnn1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(embedding)
    cnn2 = Conv1D(filters=64, kernel_size=4, strides=1, padding='valid', activation='relu')(embedding)
    cnn3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu')(embedding)
    concatenate = Concatenate(axis=1)([cnn1, cnn2, cnn3])
    maxPool1D = MaxPool1D(pool_size=2)(concatenate)
    flatten = Flatten()(maxPool1D)
    dropout = Dropout(0.2)(flatten)
    output = Dense(2, activation='softmax')(dropout)
    model = Model(inputs=[input], outputs=[output])

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

def do_cnn_doc2vec_2d(trainX, testX, trainY, testY, max_features, max_document_length):
    '''
        cnn and doc2vec
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    trainX = trainX.reshape([-1, max_features, max_document_length, 1])
    testX = testX.reshape([-1, max_features, max_document_length, 1])

    # Building convolutional network
    input = Input(shape=[None, max_features, max_document_length, 1])
    conv2D1 = Conv2D(filters=16, kernel_size=3, activation='relu')(input)
    maxPool2D1 = MaxPool2D(2)(conv2D1)
    batchNormalization1 = BatchNormalization()(maxPool2D1)
    conv2D2 = Conv2D(filters=32, kernel_size=3, activation='relu')(batchNormalization1)
    maxPool2D2 = MaxPool2D(2)(conv2D2)
    batchNormalization2 = BatchNormalization()(maxPool2D2)
    dense1 = Dense(128, activation='tanh')(batchNormalization2)
    dropout1 = Dropout(0.8)(dense1)
    dense2 = Dense(256, activation='tanh')(dropout1)
    dropout2 = Dropout(0.8)(dense2)
    output = Dense(10, activation='softmax')(dropout2)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_cnn_word2vec_2d(trainX, testX, trainY, testY, max_features, max_document_length):
    '''
        cnn and word2vec
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    input = Input(shape=[None, max_features, max_document_length, 1])
    conv2D1 = Conv2D(filters=32, kernel_size=3, activation='relu')(input)
    maxPool2D1 = MaxPool2D(2)(conv2D1)
    batchNormalization1 = BatchNormalization()(maxPool2D1)
    conv2D2 = Conv2D(filters=64, kernel_size=3, activation='relu')(batchNormalization1)
    maxPool2D2 = MaxPool2D(2)(conv2D2)
    batchNormalization2 = BatchNormalization()(maxPool2D2)
    dense1 = Dense(128, activation='tanh')(batchNormalization2)
    dropout1 = Dropout(0.8)(dense1)
    dense2 = Dense(256, activation='tanh')(dropout1)
    dropout2 = Dropout(0.8)(dense2)
    output = Dense(2, activation='softmax')(dropout2)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_cnn_word2vec_2d_345(trainX, testX, trainY, testY, max_document_length, max_features):
    '''
        cnn and  word2vec  kener=3,4,5
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    print("CNN and word2vec_2d_345")
    y_test = testY

    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    imput = Input(shape=[None,max_document_length,max_features,1])
    embedding = Embedding(input_dim=1000000, output_dim=128)(imput)
    cnn1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu')(embedding)
    cnn2 = Conv2D(filters=128, kernel_size=4, strides=1, padding='valid', activation='relu')(embedding)
    cnn3 = Conv2D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu')(embedding)

    cnn = Concatenate(axis=1)([cnn1, cnn2, cnn3])

    maxPool2D = MaxPool2D(pool_size=2)(cnn)
    flatten = Flatten()(maxPool2D)
    dropout = Dropout(0.8)(flatten)
    output = Dense(2, activation='sigmoid')(dropout)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_cnn_word2vec(trainX, testX, trainY, testY, max_document_length, max_features):
    '''
        cnn and word2vec
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    print("CNN and word2vec")
    y_test = testY
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    imput = Input(shape=[None, max_document_length, max_features, 1])
    embedding = Embedding(input_dim=1000000, output_dim=128)(imput)
    cnn1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu')(embedding)
    cnn2 = Conv2D(filters=128, kernel_size=4, strides=1, padding='valid', activation='relu')(embedding)
    cnn3 = Conv2D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu')(embedding)

    cnn = Concatenate(axis=1)([cnn1, cnn2, cnn3])

    maxPool2D = MaxPool2D(pool_size=2)(cnn)
    flatten = Flatten()(maxPool2D)
    dropout = Dropout(0.8)(flatten)
    output = Dense(2, activation='sigmoid')(dropout)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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


def do_cnn_doc2vec(trainX, testX, trainY, testY, max_document_length, max_features):
    '''
        cnn and doc2vec
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    print("CNN and doc2vec")
    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    imput = Input(shape=[None, max_document_length, max_features, 1])
    embedding = Embedding(input_dim=1000000, output_dim=128)(imput)
    cnn1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu')(embedding)
    cnn2 = Conv2D(filters=128, kernel_size=4, strides=1, padding='valid', activation='relu')(embedding)
    cnn3 = Conv2D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu')(embedding)

    cnn = Concatenate(axis=1)([cnn1, cnn2, cnn3])

    maxPool2D = MaxPool2D(pool_size=2)(cnn)
    flatten = Flatten()(maxPool2D)
    dropout = Dropout(0.8)(flatten)
    output = Dense(2, activation='sigmoid')(dropout)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_rnn_wordbag(trainX, testX, trainY, testY, max_features, max_document_length):
    '''
        rnn and wordbag
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    print("RNN and wordbag")
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    input = Input([max_features, ])
    emb = Embedding(input_dim=10240000, output_dim=128)(input)
    net = LSTM(128)(emb)
    dropout = Dropout(0.5)(net)
    dense = Dense(2, activation='softmax')(dropout)
    model = Model(inputs=[input], outputs=[dense])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_rnn_word2vec(trainX, testX, trainY, testY, max_features):
    '''
        rnn and word2vec
    Args:
        trainX:
        testX:
        trainY:
        testY:

    Returns:

    '''
    print("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    input = Input([max_features, ])
    emb = Embedding(input_dim=10240000, output_dim=128)(input)
    net = LSTM(128)(emb)
    dropout = Dropout(0.5)(net)
    dense = Dense(2, activation='softmax')(dropout)
    model = Model(inputs=[input], outputs=[dense])

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
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

def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    '''
        dnn and wordbag
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("MLP and wordbag")

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

def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
    '''
        dnn and doc2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
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

def do_dnn_word2vec(x_train, x_test, y_train, y_test):
    '''
        dnn and word2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("MLP and word2vec")
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_xgboost_word2vec(x_train, x_test, y_train, y_test):
    '''
        xgboost and word2vec
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("xgboost and word2vec")
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_xgboost_wordbag(x_train, x_test, y_train, y_test):
    '''
        xgboost and wordbag
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    print("xgboost and wordbag")
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def  get_features_by_word2vec_cnn_1d(max_features, word2ver_bin):
    '''
        for 1d mdoel
    Returns:

    '''
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    x= x_train+x_test
    cores = multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)
    min_max_scaler = preprocessing.MinMaxScaler()

    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = min_max_scaler.fit_transform(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = min_max_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

def  get_features_by_word2vec_cnn_2d(max_features, max_document_length, word2ver_bin):
    '''
        for 2d model
    Returns:

    '''
    x_train, x_test, y_train, y_test = load_all_files()

    x_train_vecs = []
    x_test_vecs = []

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    x = x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)


    #x_train_vec=np.zeros((max_document_length,max_features))
    #x_test_vec=np.zeros((max_document_length, max_features))
    """
    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = min_max_scaler.fit_transform(x_train)
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = min_max_scaler.transform(x_test)
    vec += imdb_w2v[word].reshape((1, size))
    """
    #x_train = np.concatenate([buildWordVector_2d(model, z, max_features) for z in x_train])
    x_all=np.zeros((1,max_features))
    for sms in x_train:
        sms = sms[:max_document_length]
        #print sms
        x_train_vec = np.zeros((max_document_length, max_features))
        for i,w in enumerate(sms):
            vec = model[w].reshape((1, max_features))
            x_train_vec[i-1] = vec.copy()
            #x_all=np.concatenate((x_all,vec))
        x_train_vecs.append(x_train_vec)
        #print x_train_vec.shape
    for sms in x_test:
        sms=sms[:max_document_length]
        #print sms
        x_test_vec = np.zeros((max_document_length, max_features))
        for i,w in enumerate(sms):
            vec = model[w].reshape((1, max_features))
            x_test_vec[i-1] = vec.copy()
            #x_all.append(vec)
        x_test_vecs.append(x_test_vec)

    #print x_train
    #print x_all
    min_max_scaler = preprocessing.MinMaxScaler()
    print("fix min_max_scaler")
    x_train_2d = np.concatenate([z for z in x_train_vecs])
    min_max_scaler.fit(x_train_2d)
    x_train = np.concatenate([min_max_scaler.transform(i) for i in x_train_vecs])
    x_test = np.concatenate([min_max_scaler.transform(i) for i in x_test_vecs])
    x_train = x_train.reshape([-1, max_document_length, max_features, 1])
    x_test = x_test.reshape([-1, max_document_length, max_features, 1])

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    print("Hello sms")
    # x_train, x_test, y_train, y_test = get_features_by_wordbag()
    # x_train, x_test, y_train, y_test = get_features_by_wordbag_tfidf()
    # x_train, x_test, y_train, y_test = get_features_by_ngram()
    x_train, x_test, y_train, y_test = get_features_by_tf()
    # x_train, x_test, y_train, y_test = get_features_by_doc2vec()
    # x_train, x_test, y_train, y_test = get_features_by_word2vec()
    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(y_train.shape)

