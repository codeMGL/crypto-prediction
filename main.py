import time
import numpy as np

from utils.neural_network import NeuralNetwork
import utils.functions as utils

# xs: (variables, data_length)     --> (6, 1438)
# ys: (data_length,)               --> (1438,)
# w: (w1, variables), (w1, output) -->
# b: (w1, 1), (w2, 1), ?(1, 1)     -->

# ---------------------------------------------------------------------
#                       SOME TO-DOs and notes
# To add new ETH prices, we need to preserve X_min and X_max
# Graph errors and predictions
# Export as JSON (even to re-download it later?)
# Not currently storing "Date" on 'data' list
# Add 'general info' prints to main.py
# REFACTOR: all_headers vs. model_headers
# Guardar modelos en .csv (JSON)
# [Unir y unificar datos en un solo archivo]


X_train, X_test, Y_train, Y_test = [], [], [], []

# --- Hyperparameters ---
# Percentage of data used in training
DATA_SPLIT = 70  # 70%
learning_rate = 0.004  # 0.03, 0.005 # Prev: 0.005
training_steps = 3000  # 100, 5000   # Prev: 3000


def main():
    # model_name = "good_MSE_0.05_MAE_0.07_1.txt"
    model_name = "med_mid_2.txt"
    print("-" * 10)
    # print("- Poder re-entrenar -")
    # print("Datos de varios dias")
    print("Probar diferente arquitectura y LR (varios a la vez)")
    print("Probar otros datos")
    print("-- Comparar predictions con el precio 2025 --")
    print(" HAY OVERFITTING? -> Test loss siempre baja. Raro // (Quitar precios 2021?)")
    print("-- Comprobar que no hay data leakage --")
    print("=== GUARDAR EN GITHUB === ")
    print("Average price variation 2025: 85$ (Min: 0.2$, Max: 600$)")
    print("-" * 10, "\n")

    user_input = input(f"Do you want to load {model_name}? (Y/n) ").lower()
    # Whether to create a new model or test a saved one
    if user_input == "n":
        # --- TRAIN NEW MODEL ---
        # --- Loading and parsing data ---
        # print("-- Loading and parsing data --")
        X_train, X_test, Y_train, Y_test = utils.loadData(DATA_SPLIT)
        # print("Xs", X_train.shape, X_test.shape, "Ys", Y_train.shape, Y_test.shape)

        # --- Normalizing train data ---
        # Returning the min and max prices to reverse normalisation in MAPE
        X_train_norm, Y_train_norm, norm_price = utils.normalizeTrainData(
            X_train, Y_train
        )
        X_test_norm, Y_test_norm = utils.normalizeTestData(X_test, Y_test, norm_price)

        # --- Creating the model ---
        # layers = [X_train.shape[0], 256, 256, 1]
        # layers = [X_train.shape[0], 64, 32, 1]
        # layers = [X_train.shape[0], 64, 64, 32, 1]
        layers = [X_train.shape[0], 32, 1] # Overfitting/Data leakage test
        layers = [X_train.shape[0], 32, 16, 1]
        # BUENA  MSE: 0.0007 MAE: 0.0220  | learning rate = 0.005, layers = [5, 64, 64, 32, 1]

        data_num = X_train.shape[1]
        model = NeuralNetwork(layers, data_num, learning_rate)
        print(
            f"\nTRAINING: {training_steps} steps, learning rate = {learning_rate}, layers = {layers}"
        )

        # --- Training the model ---
        t0 = time.time()
        model.train(
            X_train_norm,
            Y_train_norm,
            X_test_norm,
            Y_test_norm,
            norm_price,
            training_steps,
        )
        t_f = time.time()
        print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")

        # --- Predicting new data ---
        print("\nTEST WITH 2025 DATA:")
        X, Y = utils.loadRealData("data/ETH_1D_CMC.csv")
        X_norm, Y_norm = utils.normalizeTestData(X, Y, norm_price)
        # Predictions
        model.testRealData(X_norm, Y_norm, norm_price)

        # --- Option to save the model ---
        print("MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)")
        save = input("\nDo you want to save the model: (Y/n) ")
        if save.lower() == "y":
            name = input("Write the model name (without the extension): ")
            model.saveModel(name, norm_price)

        # Option to make a second training process ---
        double_train = input(
            f"\nWrite the new [low] learning rate used ({model.lr}) or 'n': "
        )
        if double_train != "n":
            model.lr = float(double_train)
            new_steps = int(input("Write the number of training steps [high]: "))
            print(model)
            t0 = time.time()
            model.train(
                X_train_norm,
                Y_train_norm,
                X_test_norm,
                Y_test_norm,
                norm_price,
                new_steps,
            )
            t_f = time.time()
            print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")

            print("\nTEST WITH 2025 DATA:")
            X, Y = utils.loadRealData("data/ETH_1D_CMC.csv")
            X_norm, Y_norm = utils.normalizeTestData(X, Y, norm_price)
            model.testRealData(X_norm, Y_norm, norm_price)

        # --- Option to save the new model ---
        print("MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)")
        save = input("\nDo you want to save the NEW model: (Y/n) ")
        if save.lower() == "y":
            name = input("Write the model name (without the extension): ")
            model.saveModel(name, norm_price)

    else:
        # --- LOADING A SAVED MODEL ---
        # --- Loading data ---
        # Model loaded from the text file (Weights, biases, layers, learning rate...)
        loaded_params = utils.loadModel("models/" + model_name)
        # print("Params:", loaded_params)

        # --- Initializing data and replacing by loaded data
        model = NeuralNetwork(loaded_params["layers"], loaded_params["num"])
        print("Printing loaded model:\n", model)
        # model.loadData(model_params["Ws and Bs"]) # ERROR???

        # --- Predicting real data ---
        # Loading and normalizing 2025 data
        X, Y = utils.loadRealData("data/ETH_1D_CMC.csv")
        X_norm, Y_norm = utils.normalizeTestData(X, Y, loaded_params["norm_price"])
        # Predictions
        print("\nPREDICTIONS WITH 2025 DATA:")
        print(
            f"\nloaded_params {loaded_params.keys()}\nLoaded[norm_price]: {loaded_params["norm_price"]}\n"
        )
        model.testRealData(X_norm, Y_norm, loaded_params["norm_price"])
        print()

        # # Predict just one price
        # num = None
        # while num != -1:
        #     num = int(input(f"Choose an index (0-{X_norm.shape[1]}) to predict its price: "))
        #     X_0 = X_norm[:, num]  # Every element in the first column
        #     Y_0 = Y_norm[num]  # The first element in Y
        #     X_0 = np.reshape(X_0, (X_norm.shape[0], 1))
        #     Y_0 = np.reshape(Y_0, (1, 1))
        #     model.testRealData(X_0, Y_0, loaded_params["norm_price"])


if __name__ == "__main__":
    main()
