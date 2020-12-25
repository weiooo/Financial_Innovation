#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import itertools
import numpy as np

def get_cnn_model(params):
    model = Sequential()

    model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(10, 10, 4)))
    model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(10, 10, 4)))
    #model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
    model.add(Dropout(0.25))

    model.add(Conv2D(48, (3,3), activation="relu", padding="same", data_format="channels_last"))
    model.add(Conv2D(48, (3,3), activation="relu", padding="same", data_format="channels_last"))
    #model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    return model

def train_model(params, data):
    model = get_cnn_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_gasf'], y=data['train_label_arr'],
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    return (model, hist)


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def mnist_cnn_main(data):
    # data
    PARAMS = {}
    PARAMS['classes'] = 3
    PARAMS['lr'] = 0.01
    PARAMS['epochs'] = 10
    PARAMS['batch_size'] = 64
    PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])

    x_train, y_train, x_test, y_test = data['train_gasf'], data['train_label'][:, 0], data['test_gasf'], data['test_label'][:, 0]
    # train cnn model
    model, hist = train_model(PARAMS, data)
    # train & test result
    scores = model.evaluate(data['test_gasf'], data['test_label_arr'], verbose=0)
    print('CNN test accuracy:', scores[1])

    # Plot Confusion Matrix
    # get train & test predictions
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)

    # confusion matrix
    train_result_cm = confusion_matrix(data['train_label'][:, 0], train_pred, labels=range(3))
    test_result_cm = confusion_matrix(data['test_label'][:, 0], test_pred, labels=range(3))

    plot_confusion_matrix(train_result_cm, range(0, 3))
    plot_confusion_matrix(test_result_cm, range(0, 3))
    
if __name__ == "__main__":
    # read the data and transform to certain type(dict)
    file = 'C:/Users/user.LAPTOP-EQJHNTF3/Desktop/1091/Financial_Vision/HW3/data/eurusd_2010_2012_2rulebase.csv'
    data_dict1 = data_csv2dict(file, datatype = 'ohlc')
    data_dict2 = data_csv2dict(file, datatype = 'curl')
    
    # Compare two different data feature data
    mnist_cnn_main(data_dict1)  # OHLC
    mnist_cnn_main(data_dict2)  # CURL

