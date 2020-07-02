from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import tensorflow.keras as keras

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import dispaly_res


max_features = 5000
max_document_length = 100


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    '''词袋模型构建的贝叶斯

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(dispaly_res.get_accuracy_score(y_test, y_pred))
    print(dispaly_res.get_confusion_matrix(y_test, y_pred))

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    '''#词袋模型构建的支持向量机

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(dispaly_res.get_accuracy_score(y_test, y_pred))
    print(dispaly_res.get_confusion_matrix(y_test, y_pred))

def do_cnn_wordbag(trainX, testX, trainY, testY):
    '''test cnn resoult

    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return:
    '''
    global max_document_length
    global max_features
    # list->numpy
    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    # padding
    pad_sequences(trainX, maxlen=max_features)
    pad_sequences(testX, maxlen=max_features)
    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)
    # model build
    model = keras.Sequential([
        layers.Embedding(input_dim=max_features, output_dim=128, input_length=max_features),
        layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),
        layers.MaxPool1D(2, padding='valid'),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    # 训练
    history = model.fit(trainX, trainY, batch_size=32, epochs=5, validation_split=0.1)
    dispaly_res.plot_graphs(history, 'accuracy')

def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    '''使用词袋模型+dnn 完成垃圾邮件分类任务  # 基于词袋模型的dnn MLP  参数？  形状=3k+5k

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    # list->numpy
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # 构建模型
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(5000,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # 配置模型
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    # 训练
    model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1)
    # 模型评估
    model.evaluate(x_test, y_test)
    # 展示
    print(model.metrics_names)

# def do_cnn():
#     num_features = 3000
#     sequence_length = 300
#     embedding_dimension = 100
#     (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
#     x_train = pad_sequences(x_train, maxlen=sequence_length)
#     x_test = pad_sequences(x_test, maxlen=sequence_length)
#     print(x_train.shape)
#     print(x_test.shape)
#     print(y_train.shape)
#     print(y_test.shape)
#
#     model = keras.Sequential([
#         layers.Embedding(input_dim=num_features, output_dim=embedding_dimension, input_length=sequence_length),
#         layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),
#         layers.MaxPool1D(2, padding='valid'),
#         layers.Flatten(),
#         layers.Dense(10, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=keras.optimizers.Adam(1e-3),
#                   loss=keras.losses.BinaryCrossentropy(),
#                   metrics=['accuracy'])
#
#     model.summary()
#     history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.legend(['training', 'valiation'], loc='upper left')
#     plt.show()

# def do_rnn():
#     import tensorflow_datasets as tfds
#     dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
#     train_dataset, test_dataset = dataset['train'], dataset['test']
#     tokenizer = info.features['text'].encoder
#
#     BUFFER_SIZE = 10000
#     BATCH_SIZE = 64
#
#     train_dataset = train_dataset.shuffle(BUFFER_SIZE)
#     train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
#     test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)
#     # 固定维度
#     # def get_model():
#     #     inputs = tf.keras.Input((1240,))
#     #     emb = tf.keras.layers.Embedding(tokenizer.vocab_size, 64)(inputs)
#     #     h1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(emb)
#     #     h1 = tf.keras.layers.Dense(64, activation='relu')(h1)
#     #     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h1)
#     #     model = tf.keras.Model(inputs, outputs)
#     #     return model
#     # 序列化 句子变长
#     model = tf.keras.Sequential([
#             tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
#     plot_graphs(history, 'accuracy')
#     test_loss, test_acc = model.evaluate(test_dataset)
#     print('test loss: ', test_loss)
#     print('test acc: ', test_acc)

