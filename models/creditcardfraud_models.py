# -*- coding:utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def do_xgboost(x_train, x_test, y_train, y_test):
    '''
       model = xgboost
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    do_metrics(y_test, y_pred)

def do_mlp(x_train, x_test, y_train, y_test):
    '''
        mlp
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

def do_nb(x_train, x_test, y_train, y_test):
    '''
        nb
    Args:
        x_train:
        x_test:
        y_train:
        y_test:

    Returns:

    '''
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)

def do_metrics(y_test,y_pred):
    '''

    Args:
        y_test:
        y_pred:

    Returns:

    '''
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test,y_pred))

def run_1():
    '''

    Returns:

    '''
    x_train, x_test, y_train, y_test=get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)
    do_mlp(x_train, x_test, y_train, y_test)
    do_nb(x_train, x_test, y_train, y_test)

def run_2():
    '''

    Returns:

    '''
    x_train, x_test, y_train, y_test=get_feature_undersampling()
    print("XGBoost")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("mlp")
    do_mlp(x_train, x_test, y_train, y_test)
    print("nb")
    do_nb(x_train, x_test, y_train, y_test)

def run_3():
    '''

    Returns:

    '''
    x_train, x_test, y_train, y_test = get_feature_upsampling()
    print("XGBoost")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("mlp")
    do_mlp(x_train, x_test, y_train, y_test)
    print("nb")
    do_nb(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_feature_upsampling()
    print(x_train)
    print(x_train.shape)
    # 特征提取使用标准化
    # run_1()
    # 特征提取使用标准化&降采样
    # run_2()
    # 特征提取使用标准化&过采样
    # run_3()
#print y_train

