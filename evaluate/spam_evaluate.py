import numpy as np
from models import spam_models
from utils import dispaly_res
from tensorflow.keras.preprocessing.sequence import pad_sequences



def evaluate_nb(x_test, y_test):
    '''

    :param x_test:
    :param y_test:
    :return:
    '''
    nb = model(x_test, y_test)     # 加载保存的模型
    y_pred = nb.predict(x_test)
    print(dispaly_res.get_accuracy_score(y_test, y_pred))
    print(dispaly_res.get_confusion_matrix(y_test, y_pred))

def evaluate_svm(x_test, y_test):
    '''

    :param x_test:
    :param y_test:
    :return:
    '''
    svm = model(x_test, y_test)     # 加载保存的模型
    y_pred = svm.predict(x_test)
    print(dispaly_res.get_accuracy_score(y_test, y_pred))
    print(dispaly_res.get_confusion_matrix(y_test, y_pred))

def evaluate_dnn(x_test, y_test):
    '''

    :param x_test:
    :param y_test:
    :return:
    '''
    # list->numpy
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    dnn = model(x_test, y_test)    # 加载保存的模型
    # 模型评估
    dnn.evaluate(x_test, y_test)
    # 展示
    print(dnn.metrics_names)

def evaluate_cnn(x_test, y_test):
    '''

    :param x_test:
    :param y_test:
    :return:
    '''
    # list->numpy
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # padding
    pad_sequences(x_test, maxlen=5000)
    cnn = model(x_test, y_test)    # 加载保存的模型
    # 模型评估
    dnn.evaluate(x_test, y_test)
    # 展示
    print(dnn.metrics_names)