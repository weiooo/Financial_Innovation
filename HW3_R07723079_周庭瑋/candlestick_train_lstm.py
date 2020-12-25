#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import confusion_matrix
import pickle
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import itertools

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

def lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes):
    x_train = x_train.reshape(-1, n_step, n_input)
    x_test = x_test.reshape(-1, n_step, n_input)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return (x_train, x_test, y_train, y_test)

def lstm_model(x_train, n_input, n_step, n_hidden, n_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    return model

def train_lstm(model, x_train, y_train, x_test, y_test, 
        learning_rate, training_iters, batch_size):
    adam = Adam(lr=learning_rate)
    model.summary()
    model.compile(optimizer=adam,
        loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,
        batch_size=batch_size, epochs=training_iters,
        verbose=1, validation_data=(x_test, y_test))

def mnist_lstm_main(data):
    # training parameters
    learning_rate = 0.001
    training_iters = 10
    batch_size = 128

    # model parameters
    n_input = 40
    n_step = 10
    n_hidden = 256
    n_classes = 3

    x_train, y_train, x_test, y_test = data['train_gasf'], data['train_label'][:, 0], data['test_gasf'], data['test_label'][:, 0]
    x_train, x_test, y_train, y_test = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    model = lstm_model(x_train, n_input, n_step, n_hidden, n_classes)
    train_lstm(model, x_train, y_train, x_test, y_test, learning_rate, training_iters, batch_size)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('LSTM test accuracy:', scores[1])

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
    mnist_lstm_main(data_dict1)  # data input: OHLC
    mnist_lstm_main(data_dict2)  # data input: CURL

