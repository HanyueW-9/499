import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def TN(y_true,y_pre):
    return np.sum((y_true==0) & (y_pre==0))
def FP(y_true,y_pre):
    return np.sum((y_true==0) & (y_pre==1))
def FN(y_true,y_pre):
    return np.sum((y_true==1) & (y_pre==0))
def TP(y_true,y_pre):
    return np.sum((y_true==1) & (y_pre==1))
#混淆矩阵的定义
def confusion_matrix(y_true,y_pre):
    return np.array([
        [TN(y_true,y_pre),FP(y_true,y_pre)],
        [FN(y_true,y_pre),TP(y_true,y_pre)]
    ])
#精准率
def precision(y_true,y_pre):
    try:
        return TP(y_true,y_pre)/(FP(y_true,y_pre)+TP(y_true,y_pre))
    except:
        return 0.0
#召回率
def recall(y_true,y_pre):
    try:
        return TP(y_true,y_pre)/(FN(y_true,y_pre)+TP(y_true,y_pre))
    except:
        return 0.0

def model(train_x_scaled,y_train,valid_x_scaled,y_valid,test_x_scaled,y_test,hidden_layer,epoch,dropout_rate):
    model = Sequential()  # 层次模型
    model.add(Dense(hidden_layer, activation='relu', input_dim=train_x_scaled.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    history1 = model.fit(train_x_scaled, y_train,
                     epochs=epoch,validation_data=(valid_x_scaled, y_valid),steps_per_epoch=100,validation_steps=200)
    return model,history1.history['val_loss']

if __name__ == "__main__":
    # Load the spam data set
    spamdata = pd.read_csv('spambase.data', sep=",", header=None)

    attributeNames = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam_or_not']
    spamdata.columns = attributeNames

    # Encode class name with dict
    classLabels  = sorted(set(spamdata['spam_or_not']))
    classNames = ['not_spam','spam']
    classDict = dict(zip(classNames, range(3)))

    # Get the class values
    y = np.mat(spamdata.iloc[:,-1]).T #matrix
    y_array = np.array(spamdata.iloc[:,-1]).T
    
    # Preallocate memory, set data to matrix X
    X = spamdata[attributeNames[:-1]].as_matrix()

    # Compute values of N, M and C.
    N = spamdata.shape[0]
    M = spamdata[attributeNames[:-1]].shape[1]
    C = len(classNames)

    #  Scale the input matrix
    X = (X - np.ones((N,1))*X.mean(0))/X.std(0) # normalize
    X = np.log10((X - np.mean(X,axis = 0))/np.std(X,axis = 0)+1) # add 1 to avoid zeros and apply log

    # Divide the data into 80% train, 20% test observations
    train_X,test_X, train_y, test_y= train_test_split(X,y, train_size=0.8, test_size=0.2)
    print(train_X.shape)
    print(test_X.shape)
    # Divide the data into 50% train, 50% valid observations
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, train_size=0.5, test_size=0.5)

    # train
    model1,loss1 = model(train_X,train_y,valid_X,valid_y,test_X,test_y,20,100,0.5)
    model2, loss2 = model(train_X, train_y, valid_X, valid_y, test_X, test_y, 20, 100, 0.3)
    model3,loss3 = model(train_X, train_y, valid_X, valid_y, test_X, test_y, 20,100,0.2)
    model4,loss4 = model(train_X, train_y, valid_X, valid_y, test_X, test_y, 20,100,0.1)
    #
    # #
    plt.plot(loss1[50:], c='r', linestyle='--')
    plt.plot(loss2[50:], c='y', linestyle='-.')
    plt.plot(loss3[50:], c='b', linestyle=':')
    plt.plot(loss4[50:], c='g', linestyle='-.')
    plt.legend(['dropout-0.5', 'dropout-0.3', 'dropout-0.2','dropout-0.1'])
    plt.show()
  
    best_parameter_value = 0.1
    model1,loss1 = model(train_X,train_y,valid_X,valid_y,test_X,test_y,20,100,best_parameter_value)
    
    pre_y = model1.predict_classes(test_X)
    print(pre_y.shape)
    print(test_y.shape)
    print(accuracy_score(test_y, pre_y))

    print(classification_report(test_y, pre_y))
    print(precision(test_y, pre_y))
    print(recall(test_y, pre_y))




