import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re


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
