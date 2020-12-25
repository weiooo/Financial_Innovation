import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

import keras
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix
from keras.utils import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import itertools

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    
    plt.title("Train History")
    if train == 'accuracy':
        plt.legend([train, validation], loc='lower right', shadow=True)
    else:
        plt.legend([train, validation], loc='upper right', shadow=True)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()

def cnn_preprocess(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, x_test, y_train, y_test)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
    model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation="relu", padding="same", data_format="channels_last"))
    model.add(Conv2D(64, (3,3), activation="relu", padding="same", data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model

def trainning(model, x_train, y_train, x_test, y_test, 
              learning_rate, training_iters, batch_size):
    adam = Adam(lr=learning_rate)
    model.summary()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    train_history = model.fit(x_train, y_train,
              batch_size=batch_size, epochs=training_iters,
              verbose=1, validation_data=(x_test, y_test))
    return train_history
    
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


def mnist_cnn_main():
    # training parameters
    learning_rate = 0.001
    training_iters = 10
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = cnn_preprocess(x_train, x_test, y_train, y_test)

    model = cnn_model()
    train_history = trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('CNN test accuracy:', scores[1])
    
    # Plot training result with increasing epoch
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    
    # Plot Confusion Matrix
    # get train & test predictions
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)

    # confusion matrix
    train_result_cm = confusion_matrix(y_train, train_pred, labels=range(10))
    test_result_cm = confusion_matrix(y_test, test_pred, labels=range(10))

    plot_confusion_matrix(train_result_cm, range(0, 9))
    plot_confusion_matrix(test_result_cm, range(0, 9))


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


def lstm_model(n_input, n_step, n_hidden, n_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model

def mnist_lstm_main():
    # training parameters
    learning_rate = 0.001
    training_iters = 10
    batch_size = 128

    # model parameters
    n_input = 28
    n_step = 28
    n_hidden = 256
    n_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    model = lstm_model(n_input, n_step, n_hidden, n_classes)
    train_history = trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('LSTM test accuracy:', scores[1])
    
    # Plot training result with increasing epoch
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    
    # Plot Confusion Matrix
    # get train & test predictions
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)

    # confusion matrix
    train_result_cm = confusion_matrix(y_train, train_pred, labels=range(10))
    test_result_cm = confusion_matrix(y_test, test_pred, labels=range(10))

    plot_confusion_matrix(train_result_cm, range(0, 9))
    plot_confusion_matrix(test_result_cm, range(0, 9))

if __name__ == "__main__":
    # CNN Model
    mnist_cnn_main()
    # LSTM Model
    mnist_lstm_main()

