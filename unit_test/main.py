from data_pro import spamClassification_pro
from models import spam_models
from evaluate import spam_evaluate

x_train, x_test, y_train, y_test = spamClassification_pro.getxAndy()
# 贝叶斯
# spam_models.do_nb_wordbag(x_train, x_test, y_train, y_test)
# 支持向量机
# spam_models.do_svm_wordbag(x_train, x_test, y_train, y_test)
# MLP dnn
# spam_models.do_dnn_wordbag(x_train, x_test, y_train, y_test)
# CNN
# spam_models.do_cnn_wordbag(x_train, x_test, y_train, y_test)