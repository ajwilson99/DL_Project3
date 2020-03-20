import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Reshape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import mse, binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys, time
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')  # This code was developed on a Linux device, so the back end needed to be changed in order
# to display plots. If using Windows this line may be commented out.


# Callback class for recording epoch times (Taken from https://stackoverflow.com/a/43186440)
class TimeHistory(Callback):
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

def sampling(args):  # Taken from https://keras.io/examples/variational_autoencoder/
    """ Reparameterization trick by sampling from an isotropic unit Gaussian

    # Arguments
        args (tensor): mean and Log of variance of Q(z|X)

    # Returns
        z (tensor): sampled Latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean * K.exp(0.5 * z_log_var) * epsilon


def task_1_fully_connected(data, parameters):

    # Unpack data
    x_train = data['x_train']
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    y_test = data['y_test']

    # Create model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=784, activation='tanh'),
        Dense(units=512, activation='sigmoid'),
        Dense(units=100, activation='linear'),
        Dense(units=10, activation='softmax')
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

def task_2_small_convolutional(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1)
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = data['y_test']

    model = Sequential([
        Conv2D(filters=40, kernel_size=5,
                            activation='relu', strides=1, padding='valid', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Flatten(),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
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

    test_acc = 100 * len(np.where(y_pred == y_test)[0]) / len(y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, normalize='true')

    print("Test accuracy: {}%\n".format(test_acc))
    print("Confusion Matrix: \n{}".format(confusion_mat))

    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'])
    plt.grid()
    plt.title('Epoch-Loss Plot - Task 2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(times / 60, train_history.history['loss'])
    plt.grid()
    plt.title('Time-Loss Plot - Task 2')
    plt.xlabel('Time (min)')
    plt.ylabel('Loss')
    plt.show()


def task_3_bigger_convolutional(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1)
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = data['y_test']

    model = Sequential([
        Conv2D(filters=48, kernel_size=3,
                            activation='relu', strides=1, padding='valid', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Conv2D(filters=96, kernel_size=3,
                            activation='relu', strides=1, padding='valid'),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Flatten(),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
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

    test_acc = 100 * len(np.where(y_pred == y_test)[0]) / len(y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, normalize='true')

    print("Test accuracy: {}%\n".format(test_acc))
    print("Confusion Matrix: \n{}".format(confusion_mat))

    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'])
    plt.grid()
    plt.title('Epoch-Loss Plot - Task 3')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(times / 60, train_history.history['loss'])
    plt.grid()
    plt.title('Time-Loss Plot - Task 3')
    plt.xlabel('Time (min)')
    plt.ylabel('Loss')
    plt.show()


def task_4_custom_convolutional(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1)
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = data['y_test']

    model = Sequential([
        Conv2D(filters=48, kernel_size=3,
                            activation='relu', strides=1, padding='valid', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Conv2D(filters=96, kernel_size=3,
                            activation='relu', strides=1, padding='valid'),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Flatten(),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    ad = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='huber_loss', optimizer=ad, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
    # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)

    test_acc = 100 * len(np.where(y_pred == y_test)[0]) / len(y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, normalize='true')

    print("Test accuracy: {}%\n".format(test_acc))
    print("Confusion Matrix: \n{}".format(confusion_mat))

    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'])
    plt.grid()
    plt.title('Epoch-Loss Plot - Task 3')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(times / 60, train_history.history['loss'])
    plt.grid()
    plt.title('Time-Loss Plot - Task 3')
    plt.xlabel('Time (min)')
    plt.ylabel('Loss')
    plt.show()


def task_5_var_autoencoder(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1)
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = data['y_test']

    latent_dim = parameters["latent_dim"]
    lossfunc = 'mean_squared_error'
    learning_rate = parameters["learning_rate"]
    mini_batch_size = parameters["mini_batch_size"]

    # Build Encoder
    inputs = Input(shape=(28, 28, 1), name='encoder_input')
    c1 = Conv2D(filters=24, kernel_size=3,
                            activation='relu', strides=1, padding='valid', input_shape=(28, 28, 1))(inputs) # output shape 26x26x48
    c2 = Conv2D(filters=48, kernel_size=3,
                            activation='relu', strides=1, padding='valid')(c1)  # output shape 24x24x48
    f1 = Flatten()(c2),
    z_mean = Dense(latent_dim)(f1[0])
    z_log_var = Dense(latent_dim)(f1[0])

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
    encoder.summary()

    # Build Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(24*24*48, activation='relu',name='decoder_first_dense')(latent_inputs)
    r = Reshape(target_shape=(24, 24, 48))(x)
    dc1 = Conv2DTranspose(filters=48, kernel_size=3, activation='relu', strides=1, padding='valid')(r)
    dc2 = Conv2DTranspose(filters=24, kernel_size=3, activation='relu', strides=1, padding='valid')(dc1)
    y = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(dc2)
    decoder = Model(latent_inputs, y, name='decoder_output')
    decoder.summary()

    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE')

    # Compile VAE model
    sgd = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
    vae.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    time_callback = TimeHistory()
    train_history = vae.fit(x_train, x_train, epochs=50, batch_size=mini_batch_size, callbacks=[time_callback])

    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
    # (each time is roughly the same value, but want to plot over the entire period)

    x_pred = vae.predict(x_test) 

    # Test plot
    test_out = x_pred[527].reshape(28, 28)  # Randomly chosen value from test set - 527
    test_in = x_test[527].reshape(28, 28)

    plt.subplot(2,1,1)
    plt.imshow(test_in)
    plt.subplot(2,1,2)
    plt.imshow(test_out)

    # TODO: Plot epoch-loss, time-loss curves,
    # TODO: choose ten random latent vectors to feed into the decoder to generate plots of clothes,
    # TODO: Alter network architecture to produce different results