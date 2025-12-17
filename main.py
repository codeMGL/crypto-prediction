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
# To add new ETH prices, we need to preserve X_min and X_max --> ??

# Data 2017-2024 Starting MAPE: 400% Why???
# Graph errors and predictions without overloading data (not responding)
# Add every error (include 2025 Test) to the graph
# Add new artificial data
# Search for data leaking
# Add ALL the data the model needs to predict (includin in 2025)
""" trained_w_2025_MAPE_3_14.txt
--- Itineration 7498 ---
Normalized (Train):   MSE: 0.0007       MAE: 0.0191
Normalized (Test):    MSE: 0.0008       MAE: 0.0190
Norm (Test 2025):     MSE: 0.0175       MAE: 0.1088 --> Very high
Real scale (Train):   RMSE: 104.7396$   MAE: 77.2301$   MAPE: 3.6412%
Real scale (Test):    RMSE: 114.3909$   MAE: 76.6764$   MAPE: 3.8678%
Real (Test 2025):     RMSE: 534.6767$   MAE: 439.3730$   MAPE: 14.2091%
"""

print("-" * 10)
# print("- Poder re-entrenar -")
# print("Datos de varios dias")
print("Probar diferente arquitectura y LR (varios a la vez)")
print("Visualizar error con gráficas")
# print("Probar otros datos. Cuales?")
# print("-- Comparar predictions con el precio 2025 --")
print(" HAY OVERFITTING? -> Test loss siempre baja. Raro // (Quitar precios 2021?)")
# print("-- Comprobar que no hay data leakage --")
# print("=== GUARDAR EN GITHUB === ")
print("Average price variation 2025: 85$ (Min: 0.2$, Max: 600$)")
print("-" * 10, "\n")


# Ver notas computación para mejorar la función de activacion
# Export model as JSON, not .txt (even to re-download it later?)
# Guardar modelos en .csv (JSON)
# Not currently storing "Date" on 'data' list
# Add 'general info' prints to main.py
# REFACTOR: all_headers vs. model_headers
# [Unir y unificar datos en un solo archivo]

#                      VERSION FINAL
# --  Usar Tensorflow --
# BTC price (and volume)
# Fear and greed. Btc/Alt season
# Regularización L1/L2
# Probar StochasticGradientDescent (Actualmente usando BatchGradientDescent (entrenando con el lote completo))


X_train, X_test, Y_train, Y_test = [], [], [], []

# --- Hyperparameters ---
# Percentage of data used in training
DATA_SPLIT = 80  # 70%
learning_rate = 0.01  # 0.01, 0.005 # Prev: 0.005
training_steps = 1000  # 1000, 5000   # Prev: 3000


def main2():
    print("Running!")
    # --- TRAIN NEW MODELS ---
    X_train, X_test, Y_train, Y_test = utils.loadData(DATA_SPLIT)
    X_train_norm, Y_train_norm, norm_price = utils.normalizeTrainData(X_train, Y_train)
    X_test_norm, Y_test_norm = utils.normalizeTestData(X_test, Y_test, norm_price)
    X_2025, Y_2025 = utils.loadRealData("data/ETH_1D_CMC.csv")
    X_2025_norm, Y_2025_norm = utils.normalizeTestData(X_2025, Y_2025, norm_price)

    # --- Creating the models ---
    # More params/layers can cause overfitting
    # layers0 = [X_train.shape[0], 8, 1]
    layers1 = [X_train.shape[0], 8, 8, 1]
    layers2 = [X_train.shape[0], 64, 32, 1] # Regular
    layers3 = [X_train.shape[0], 64, 64, 32, 1] # Mejor
    # layers4 = [X_train.shape[0], 32, 16, 1]
    layers5 = [X_train.shape[0], 128, 128, 1] # Regular
    # layers6 = [X_train.shape[0], 32, 32, 16, 1]
    layers7 = [X_train.shape[0], 64, 64, 64, 1]  # Mejor
    layers8 = [X_train.shape[0], 64, 64, 1]  # Regular-Bueno
    # layers8 = [X_train.shape[0], 64, 32, 16, 1] MAL
    # layers_arr = [layers0, layers1, layers2, layers3, layers4, layers5, layers6]
    layers_arr = [layers3, layers7]
    learning_rate = [0.001, 0.0001, 0.0008, 0.01, 0.03]
    models_arr = []
    print(f"Training {len(layers_arr)} models with {len(learning_rate)} different learning rates!")
    # TRAINING 1000 steps, learning rate = 0.0008, layers = [9, 64, 64, 32, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 32, 1] MUY BUENO (MAPE: 4-6)
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 32, 1] BUENO, UN POCO DE OVERFIT

    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 64, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 64, 1] MUY BUENO, UN POCO DE OVERFIT

    data_num = X_train.shape[1]
    # Training all models, one by one
    for i in range(len(layers_arr)):
        for lr in learning_rate:
            model = NeuralNetwork(layers_arr[i], data_num, lr)
            print(f"\n======== MODEL {i} ========")
            print(
                f"TRAINING: {training_steps} steps, learning rate = {lr}, layers = {layers_arr[i]}"
            )

            # --- Training the model ---
            t0 = time.time()
            model.train(
                X_train_norm,
                Y_train_norm,
                X_test_norm,
                Y_test_norm,
                X_2025_norm,
                Y_2025_norm,
                norm_price,
                training_steps,
            )
            print("-- FINAL TEST --")
            errors = model.test(
                X_train_norm, Y_train_norm, X_test_norm, Y_test_norm, X_2025_norm, Y_2025_norm, norm_price
            )
            errors = [round(float(err), 5) for err in errors]
            t_f = time.time()
            runningTime = round((t_f - t0) / 60, 2)
            print(f"Training took {runningTime} minutes!")
            models_arr.append({"model": model, "errors": errors, "t": runningTime, "lr": lr})
    print("=" * 15, "END", "=" * 15)

    # Prints all models
    print(
        f"\nERRORS (Test data):{" " * 25}mse_norm_train, mse_norm_test, mae_norm_train, mae_norm_test, mape_train, mape_test"
    )
    for i in range(len(models_arr)):
        layers = models_arr[i]["model"].layers
        formatting = " " * 3 * int(6 - len(layers))
        print(i, layers, models_arr[i]["lr"], f"{models_arr[i]["t"]} mins", formatting, models_arr[i]["errors"])
    print(models_arr)
    # --- Option to save the model ---
    print("MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)")
    index = int(input("Choose a model to save: "))
    save = input("\nDo you want to save the model: (Y/n) ")
    if save.lower() == "y":
        name = input("Write the model name (without the extension): ")
        models_arr[index]["model"].saveModel(name, norm_price)

    # Option to make a second training process ---
    double_train = input(
        f"\nWrite the new [low] learning rate used ({models_arr[index]["model"].lr}) or 'n': "
    )
    if double_train != "n":
        models_arr[index]["model"].lr = float(double_train)
        new_steps = int(input("Write the number of training steps [high]: "))
        print(models_arr[index])
        t0 = time.time()
        models_arr[index]["model"].train(
            X_train_norm,
            Y_train_norm,
            X_test_norm,
            Y_test_norm,
            X_2025_norm,
            Y_2025_norm,
            norm_price,
            new_steps,
            prevStepsCount=training_steps
        )
        t_f = time.time()
        print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")

        print("\nTEST WITH 2025 DATA:")
        models_arr[index]["model"].testRealData(X_2025_norm, Y_2025_norm, norm_price)

    # --- Option to save the new model ---
    print("MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)")
    save = input("\nDo you want to save the NEW model: (Y/n) ")
    if save.lower() == "y":
        name = input("Write the model name (without the extension): ")
        models_arr[index]["model"].saveModel(name, norm_price)


def main():
    model_name = "model_MAPE__3_4_17.txt"
    model_name = "trained_w_2025_MAPE_3_14.txt"


    user_input = input(f"Do you want to load {model_name}? (Y/n) ").lower()
    # print("USER_INPUT CHANGED!!!")
    # user_input = "n"
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
        # print(X_train, "\nX train", X_train_norm)

        # --- Loading train data (2025 prices) ---
        X_2025, Y_2025 = utils.loadRealData("data/ETH_1D_CMC.csv")
        X_2025_norm, Y_2025_norm = utils.normalizeTestData(X_2025, Y_2025, norm_price)

        # --- Creating the model ---
        # More params/layers can cause overfitting
        # layers = [X_train.shape[0], 256, 256, 1]
        # layers = [X_train.shape[0], 64, 32, 1]
        # layers = [X_train.shape[0], 32, 16, 1]
        # layers = [X_train.shape[0], 8, 8, 1]
        # layers = [X_train.shape[0], 32, 32, 16, 1]
        layers = [X_train.shape[0], 64, 64, 32, 1]

        # TRAINING 1000 steps, learning rate = 0.0008, layers = [9, 64, 64, 32, 1] BUENO
        # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 32, 1] MUY BUENO (MAPE: 4-6)
        # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 32, 1] BUENO, UN POCO DE OVERFIT

        # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 64, 1] BUENO
        # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 64, 1] MUY BUENO, UN POCO DE OVERFIT
        # Falta 0.005

        data_num = X_train.shape[1]
        model = NeuralNetwork(layers, data_num, learning_rate)
        print(
            f"------\nTRAINING: {training_steps} steps, learning rate = {learning_rate}, layers = {layers}\n------"
        )

        # --- Training the model & testing with 2025 data ---
        t0 = time.time()
        model.train(
            X_train_norm,
            Y_train_norm,
            X_test_norm,
            Y_test_norm,
            X_2025_norm,
            Y_2025_norm,
            norm_price,
            training_steps,
        )
        t_f = time.time()
        print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")
        print("=" * 15, "END", "=" * 15)

        # --- Predicting new data at the end ---
        print("\nTEST WITH 2025 DATA:")
        model.testRealData(X_2025_norm, Y_2025_norm, norm_price)

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
                X_2025_norm,
                Y_2025_norm,
                norm_price,
                new_steps,
                training_steps
            )
            t_f = time.time()
            print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")

            print("\nTEST WITH 2025 DATA:")
            model.testRealData(X_2025_norm, Y_2025_norm, norm_price)

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
        X_2025, Y_2025 = utils.loadRealData("data/ETH_1D_CMC.csv")
        X_2025_norm, Y_2025_norm = utils.normalizeTestData(
            X_2025, Y_2025, loaded_params["norm_price"]
        )
        # Predictions
        print("\nPREDICTIONS WITH 2025 DATA:")
        print(
            f"\nloaded_params {loaded_params.keys()}\nLoaded[norm_price]: {loaded_params["norm_price"]}\n"
        )
        model.testRealData(X_2025_norm, Y_2025_norm, loaded_params["norm_price"])
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
