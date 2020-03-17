import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys, time
from sklearn.metrics import confusion_matrix

# Aaron Wilson
# March, 2020

matplotlib.use('TkAgg')  # This code was developed on a Linux device, so the back end needed to be changed in order
# to display plots.


# Callback class for recording epoch times (Taken from https://stackoverflow.com/a/43186440)
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def min_max_scale_data(x_train, x_test):  # Min-max scaling of data

    maxval = np.max(np.max(x_train))
    minval = np.min(np.min(x_train))

    x_train_norm = ((x_train - minval)/(maxval - minval))

    x_test_norm = ((x_test - minval)/(maxval - minval))

    return (x_train_norm, x_test_norm)


def task_1_fully_connected(data, parameters):

    # Unpack data
    x_train = data['x_train']
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    y_test = data['y_test']

    # Create model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(784, activation='tanh'),
        keras.layers.Dense(512, activation='sigmoid'),
        keras.layers.Dense(100, activation='linear'),
        keras.layers.Dense(10, activation='softmax')
    ])

    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    sgd = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
                                        # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)

    test_acc = 100*len(np.where(y_pred == y_test)[0])/len(y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, normalize='true')

    print("Test accuracy: {}%\n".format(test_acc))
    print("Confusion Matrix: \n{}".format(confusion_mat))

    plt.subplot(1,2,1)
    plt.plot(train_history.history['loss'])
    plt.grid()
    plt.title('Epoch-Loss Plot - Task 1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.subplot(1,2,2)
    plt.plot(times/60, train_history.history['loss'])
    plt.grid()
    plt.title('Time-Loss Plot - Task 1')
    plt.xlabel('Time (min)')
    plt.ylabel('Loss')
    plt.show()
    a = 'debug'

def main():

    tasknum = int(sys.argv[1])  # Can be a number, 1 to 5
    #tasknum = 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()  # Load data

    (x_train, x_test) = min_max_scale_data(x_train, x_test)  # Normalize data

    # Package data into a single variable for passing to functions
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test
    }

    if tasknum == 1:
        parameters = {
            "learning_rate": 0.1,
            "epochs": 50,
            "mini_batch_size": 200
        }
        task_1_fully_connected(data, parameters)

    a = 1  # debug


if __name__ == "__main__":

   main()