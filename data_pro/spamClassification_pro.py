from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import codecs


max_features = 5000
max_document_length = 100


# 1.加载数据
def load_one_file(filename):
    '''
    :param filename:
    :return: mail-content
    '''
    x = ""
    with codecs.open(filename, encoding=u'utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    return x

def load_files_from_dir(rootdir):
    """
    :param rootdir:
    :return: dir-content
    """
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    '''
    :return: all-content
    '''
    ham = []
    spam = []
    for i in range(1, 2):
        path = "../../dataset/data_spam/enron%d/ham/" % i
        print("Load %s" % path)
        ham += load_files_from_dir(path)
        path = "../../dataset/data_spam/enron%d/spam/" % i
        print("Load %s" % path)
        spam += load_files_from_dir(path)
    return ham, spam

# 2.获取特征
def get_features_by_wordbag():
    '''词袋模型构建特征向量
    :return: embedding-res:shape(3000,5000)
    '''
    ham, spam = load_all_files()
    x = ham + spam
    y = [0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(decode_error='ignore', strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)
    print(vectorizer)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    return x, y

#tf_idf构建的词袋模型
def get_features_by_wordbag_tfidf():
    '''使用tf-idf构建向量层

    :return: embedding-res:shape(3000,5000)
    '''
    ham, spam = load_all_files()
    x = ham+spam
    y = [0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(binary=False, decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)
    print(vectorizer)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    print(transformer)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x, y

def getxAndy():
    '''返回样本集

    :return:  比例划分样本集
    '''
    x, y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    return x_train, x_test, y_train, y_test


