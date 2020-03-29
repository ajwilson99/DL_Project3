# Aaron Wilson
# March, 2020

import sys
from tensorflow import keras
import project_functions as pf

def main():

    tasknum = int(sys.argv[1])  # Can be a number, 1 to 5
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()  # Load data

    (x_train, x_test, maxval, minval) = pf.min_max_scale_data(x_train, x_test)  # Normalize data

    # Package data into a single variable for passing to functions
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "max": maxval,
        "min": minval
    }

    if tasknum == 1:
        parameters = {
            "learning_rate": 0.1,
            "epochs": 50,
            "mini_batch_size": 200
        }

        pf.task_1_fully_connected(data, parameters)

    elif tasknum == 2:
        parameters = {
            "learning_rate": 0.1,
            "epochs": 50,
            "mini_batch_size": 200,
        }
        pf.task_2_small_convolutional(data, parameters)

    elif tasknum == 3:
        parameters = {
            "learning_rate": 0.1,
            "epochs": 50,
            "mini_batch_size": 200,
        }
        pf.task_3_bigger_convolutional(data, parameters)

    elif tasknum == 4:
        # Custom CNN: Added a Conv2D layer with 24 filters, another Max Pooling layer,
        # momentum with gamma = 0.9, let nesterov = True.
        parameters = {
            "learning_rate": 0.1,
            "epochs": 50,
            "mini_batch_size": 200
        }
        pf.task_4_custom_convolutional(data, parameters)

    elif tasknum == 5:
        parameters = {
            "learning_rate": 0.01,
            "epochs": 50,
            "mini_batch_size": 200,
            "latent_dim": 10,
            "loss_func": "mean_squared_error" # or "binary_cross_entropy"
        }
        pf.task_5_var_autoencoder(data, parameters)

    elif tasknum == 6:
        parameters = {
            "learning_rate": 0.001,
            "epochs": 50,
            "mini_batch_size": 200,
            "latent_dim": 20,
            "loss_func": "mean_squared_error"
        }
        pf.task_6_var_autoencoder_alt(data, parameters)

if __name__ == "__main__":

   main()
