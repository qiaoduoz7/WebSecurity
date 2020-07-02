import matplotlib.pyplot as plt
from sklearn import metrics

def plot_graphs(history, string):
    '''查看训练过程  绘制图形

    :param history:
    :param string:
    :return:
    '''
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def get_accuracy_score(y_test, y_pred):
    '''返回acc

    :param y_test:
    :param y_pred:
    :return: acc
    '''
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

def get_confusion_matrix(y_test, y_pred):
    '''  cmputer confusion_matrix

    :param y_test:
    :param y_pred:
    :return: confusion_matrix
    '''
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return confusion_matrix