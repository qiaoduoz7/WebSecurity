from data_pro import spamClassification_pro
from models import spam_models
from evaluate import spam_evaluate

# x_train, x_test, y_train, y_test = spamClassification_pro.getxAndy()
# 贝叶斯
# spam_models.do_nb_wordbag(x_train, x_test, y_train, y_test)
# 支持向量机
# spam_models.do_svm_wordbag(x_train, x_test, y_train, y_test)
# MLP dnn
# spam_models.do_dnn_wordbag(x_train, x_test, y_train, y_test)
# CNN
# spam_models.do_cnn_wordbag(x_train, x_test, y_train, y_test)

# 读取数据
# x_train, x_test, y_train, y_test = load_all_files()
# 获取特征（词袋模型）
# x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag()
# x_train = comment_data_pro.listConvertNumpy(x_train)
# x_test = comment_data_pro.listConvertNumpy(x_test)
# y_train = comment_data_pro.listConvertNumpy(y_train)
# y_test = comment_data_pro.listConvertNumpy(y_test)
# 获取特征（词袋tf_idf模型）
# x_train, x_test, y_train, y_test = comment_data_pro.get_features_by_wordbag_tfidf()
# 获取特征（word2vec模型）
# x_idx, y, embedMatrix = comment_data_pro.get_features_by_word2vec()
# 获取特征（dec2vec模型）
# x_vec, y = comment_data_pro.get_features_by_doc2vec()
# 划分比例
# x_train, x_test, y_train, y_test = comment_data_pro.getsample(x_vec, y)
# cnn and dec2vec
# do_cnn_dec2vec(x_train, x_test, y_train, y_test)
# dnn and dec2vec
# do_dnn_doc2vec(x_train, x_test, y_train, y_test)
# rnn and word2vec
# do_rnn_word2vec(x_train, x_test, y_train, y_test, embedMatrix)
# cnn and word2vec
# do_cnn_word2vec(x_train, x_test, y_train, y_test, embedMatrix)