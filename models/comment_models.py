from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from data_pro import comment_data_pro


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))




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
    x_train, x_test, y_train, y_test, embedMatrix = comment_data_pro.get_features_by_doc2vec()
    y_train = comment_data_pro.listConvertNumpy(y_train)
    y_test = comment_data_pro.listConvertNumpy(y_test)


    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag()
    # x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_doc2vec()
    do_nb_wordbag(x_train, x_test, y_train, y_test)