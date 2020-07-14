# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def get_feature():
    '''    标准化

    Returns:

    '''
    df = pd.read_csv("../../dataset/creditcardfraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))     # 标准化 映射到0-1之间
    df = df.drop(['Time', 'Amount'], axis=1)    # 删除无用字段
    y = df['Class']     # 标签字段
    #print y
    features = df.drop(['Class'], axis=1).columns    # 删除标签字段 将剩下全部字段的索引，复制给数据集作为x
    x = df[features]
    #print x
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_undersampling():
    '''
        undersampling
    Returns:

    '''
    df = pd.read_csv("../../dataset/creditcardfraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    #  获取黑样本数量以及对应的索引
    number_fraud = len(df[df.Class==1])
    # print number_fraud
    fraud_index = np.array(df[df.Class==1].index)
    # print fraud_index
    #  获取白样本的索引
    normal_index = df[df.Class==0].index
    #  随机选取与黑样本数据量相当的白样本索引
    random_choice_index = np.random.choice(normal_index, size=number_fraud, replace=False)    # np.random.choice
    # 可以从整数或一维数组里随机选取内容，
    # 并将选取结果放入n维数组中返回

    x_index = np.concatenate([fraud_index,random_choice_index])     # 重新组合新城新的样本集
    df = df.drop(['Class'], axis=1)     # x = 去掉标签集
    x = df.iloc[x_index,:]    #  提取行数据loc ix
    # print x
    y = [1]*number_fraud + [0]*number_fraud

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_undersampling_2():
    '''

    Returns:

    '''
    df = pd.read_csv("../../dataset/creditcardfraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    print("raw data")
    print(pd.value_counts(y_train))
    number_fraud = len(y_train[y_train==1])
    print(number_fraud)
    fraud_index = np.array(y_train[y_train==1].index)
    print(fraud_index)
    normal_index = y_train[y_train==0].index
    random_choice_index = np.random.choice(normal_index,size=number_fraud,replace=False)
    x_index = np.concatenate([fraud_index,random_choice_index])
    print(x_index)
    #df = df.drop(['Class'], axis=1)
    x_train_1 = x.iloc[x_index,:]
    #print x
    y_train_1 = [1]*number_fraud + [0]*number_fraud
    print("Undersampling data")
    print(pd.value_counts(y_train_1))

    return x_train_1, x_test, y_train_1, y_test

def get_feature_upsampling():
    '''

    Returns:

    '''
    df = pd.read_csv("../../dataset/creditcardfraud/creditcard.csv")
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    y = df['Class']
    features = df.drop(['Class'], axis=1).columns
    x = df[features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    print("raw data")
    print(pd.value_counts(y_train))    # 目前训练集的分布

    os = SMOTE(random_state=0)
    x_train_1, y_train_1 = os.fit_sample(x_train,y_train)
    print("Smote data")
    print(pd.value_counts(y_train_1))

    return x_train, x_test, y_train, y_test


