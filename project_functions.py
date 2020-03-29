# Aaron Wilson
# March, 2020


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

    return (x_train_norm, x_test_norm, maxval, minval)

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
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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

    # Extract parameters
    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    sgd = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
                                        # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)  # Predicted values of test set

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

    # Create model
    model = Sequential([
        Conv2D(filters=40, kernel_size=5,
                            activation='relu', strides=1, padding='valid', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Flatten(),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    # Extract parameters
    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    sgd = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
    # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)  # Predict labels of test set data

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

    # Create model
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

    # Extract parameters
    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    sgd = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
    # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)  # Predict labels for test set data

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

    # Create model
    model = Sequential([
        Conv2D(filters=96, kernel_size=3,
                            activation='relu', strides=1, padding='same', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Conv2D(filters=48, kernel_size=3,
                            activation='relu', strides=1, padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Conv2D(filters=24, kernel_size=3, activation='relu', strides=1, padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'),
        Flatten(),
        Dense(units=100, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    # Extract parameters
    learning_rate = parameters["learning_rate"]
    epochs = parameters["epochs"]
    mini_batch_size = parameters["mini_batch_size"]

    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    time_callback = TimeHistory()
    train_history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=mini_batch_size, callbacks=[time_callback])
    times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
    # (each time is roughly the same value, but want to plot over the entire period)

    y_pred = np.argmax(model.predict(x_test), axis=1)  # Predict labels for test set data

    test_acc = 100 * len(np.where(y_pred == y_test)[0]) / len(y_test)
    confusion_mat = confusion_matrix(y_test, y_pred, normalize='true')

    print("Test accuracy: {}%\n".format(test_acc))
    print("Confusion Matrix: \n{}".format(confusion_mat))

    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'])
    plt.grid()
    plt.title('Epoch-Loss Plot - Task 4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(times / 60, train_history.history['loss'])
    plt.grid()
    plt.title('Time-Loss Plot - Task 4')
    plt.xlabel('Time (min)')
    plt.ylabel('Loss')
    plt.show()


def task_5_var_autoencoder(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')
    y_test = data['y_test']

    # Extract parameters
    latent_dim = parameters["latent_dim"]
    learning_rate = parameters["learning_rate"]
    mini_batch_size = parameters["mini_batch_size"]
    epochs = parameters["epochs"]
    loss_func = parameters["loss_func"]

    # Initialize a tensorflow session for evaluating tensors
    sess = tf.Session()
    with sess.as_default():

        # Build Encoder
        inputs = Input(shape=(28, 28, 1), name='encoder_input')
        c1 = Conv2D(filters=8, kernel_size=3,
                                activation='relu', strides=2, padding='same', input_shape=(28, 28, 1))(inputs) 
        c2 = Conv2D(filters=16, kernel_size=3,
                                activation='relu', strides=2, padding='same')(c1)  
        f1 = Flatten()(c2)
        d1 = Dense(16, activation='relu')(f1)
        z_mean = Dense(latent_dim, name='z_mean')(d1)
        z_log_var = Dense(latent_dim, name='z_log_var')(d1)

        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
        encoder.summary()

        # Build Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(7*7*16, activation='relu',name='decoder_first_dense')(latent_inputs)
        r = Reshape(target_shape=(7, 7, 16))(x)
        dc1 = Conv2DTranspose(filters=16, kernel_size=3, activation='relu', strides=2, padding='same')(r)
        dc2 = Conv2DTranspose(filters=8, kernel_size=3, activation='relu', strides=2, padding='same')(dc1)
        y = Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='sigmoid')(dc2)
        decoder = Model(latent_inputs, y, name='decoder_output')
        decoder.summary()

        # Instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='VAE')
        if (loss_func == "mean_squared_error"):
            loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

        # Add KL Divergence term    
        loss *= x_train.shape[1] * x_train.shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(loss + kl_loss)
        vae.add_loss(vae_loss)

        # Compile VAE model
        sgd = keras.optimizers.SGD(lr=learning_rate)
        vae.compile(optimizer=sgd, metrics=['accuracy'])
        time_callback = TimeHistory()
        train_history = vae.fit(x_train, epochs=epochs, batch_size=mini_batch_size, callbacks=[time_callback])

        times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
        # (each time is roughly the same value, but want to plot over the entire period)
        
        # Generate 10 random latent vectors following a N(0, 1) distribution
        random_test_vectors = np.random.normal(0, 1, (10, 10))
        test_images = np.zeros((28, 28, 10))  # Initialize variable for decoder outputs based on the random latent vectors
        
        # Generate images based on ten random latent vectors
        for lv in range(0, random_test_vectors.shape[0]):
            test_images[:, :, lv] = decoder(random_test_vectors[lv].reshape(1, 10)).eval().reshape(28, 28)

        # Plotting
        test_plot_idcs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        plt.figure(0)
        for im in range(0, random_test_vectors.shape[0]):
            plt.subplot2grid((2, 5), test_plot_idcs[im])
            plt.imshow(test_images[:, :, im])

        plt.show()

        plt.subplot(1, 2, 1)
        plt.plot(train_history.history['loss'])
        plt.grid()
        plt.title('Epoch-Loss Plot - Task 5')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(times / 60, train_history.history['loss'])
        plt.grid()
        plt.title('Time-Loss Plot - Task 5')
        plt.xlabel('Time (min)')
        plt.ylabel('Loss')
        plt.show()

def task_6_var_autoencoder_alt(data, parameters):

    x_train = data['x_train']
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
    y_train = to_categorical(data['y_train'])
    x_test = data['x_test']
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')
    y_test = data['y_test']

    # Extract parameters
    latent_dim = parameters["latent_dim"]
    learning_rate = parameters["learning_rate"]
    mini_batch_size = parameters["mini_batch_size"]
    epochs = parameters["epochs"]
    loss_func = parameters["loss_func"]

    # Initialize a tensorflow session for evaluating tensors
    sess = tf.Session()
    with sess.as_default():

        # Build Encoder
        inputs = Input(shape=(28, 28, 1), name='encoder_input')
        c1 = Conv2D(filters=10, kernel_size=3,
                                activation='relu', strides=2, padding='same', input_shape=(28, 28, 1))(inputs) 
        c2 = Conv2D(filters=20, kernel_size=3,
                                activation='relu', strides=2, padding='same')(c1)  
        f1 = Flatten()(c2)
        d1 = Dense(16, activation='relu')(f1)
        z_mean = Dense(latent_dim, name='z_mean')(d1)
        z_log_var = Dense(latent_dim, name='z_log_var')(d1)

        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
        encoder.summary()

        # Build Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(7*7*20, activation='relu',name='decoder_first_dense')(latent_inputs)
        r = Reshape(target_shape=(7, 7, 20))(x)
        dc1 = Conv2DTranspose(filters=20, kernel_size=3, activation='relu', strides=2, padding='same')(r)
        dc2 = Conv2DTranspose(filters=10, kernel_size=3, activation='relu', strides=2, padding='same')(dc1)
        y = Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='sigmoid')(dc2)
        decoder = Model(latent_inputs, y, name='decoder_output')
        decoder.summary()

        # Instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='VAE')
        if (loss_func == "mean_squared_error"):
            loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

        # Add KL Divergence term    
        loss *= x_train.shape[1] * x_train.shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(loss + kl_loss)
        vae.add_loss(vae_loss)

        # Compile VAE model
        sgd = keras.optimizers.SGD(lr=learning_rate)
        vae.compile(optimizer=sgd, metrics=['accuracy'])
        time_callback = TimeHistory()
        train_history = vae.fit(x_train, epochs=epochs, batch_size=mini_batch_size, callbacks=[time_callback])

        times = np.cumsum(time_callback.times)  # Cumulative sum for plotting
        # (each time is roughly the same value, but want to plot over the entire period)
        
        # Generate 10 random latent vectors following a N(0, 1) distribution
        random_test_vectors = np.random.normal(0, 1, (10, 20))
        test_images = np.zeros((28, 28, 10))  # Initialize variable for decoder outputs based on the random latent vectors
        
        # Generate images based on ten random latent vectors
        for lv in range(0, random_test_vectors.shape[0]):
            test_images[:, :, lv] = decoder(random_test_vectors[lv].reshape(1, 20)).eval().reshape(28, 28)

        # Plotting
        test_plot_idcs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        plt.figure(0)
        for im in range(0, random_test_vectors.shape[0]):
            plt.subplot2grid((2, 5), test_plot_idcs[im])
            plt.imshow(test_images[:, :, im])

        plt.show()

        plt.subplot(1, 2, 1)
        plt.plot(train_history.history['loss'])
        plt.grid()
        plt.title('Epoch-Loss Plot - Task 5')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(times / 60, train_history.history['loss'])
        plt.grid()
        plt.title('Time-Loss Plot - Task 5')
        plt.xlabel('Time (min)')
        plt.ylabel('Loss')
        plt.show()
        
