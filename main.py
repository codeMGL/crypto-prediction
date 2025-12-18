import time
import numpy as np

from utils.neural_network import NeuralNetwork
import utils.functions as utils

from utils.graph import graph

# xs: (variables, data_length)     --> (6, 1438)
# ys: (data_length,)               --> (1438,)
# w: (w1, variables), (w1, output) -->
# b: (w1, 1), (w2, 1), ?(1, 1)     -->
""" trained_w_2025_MAPE_3_14.txt
--- Itineration 7498 ---
Normalized (Train):   MSE: 0.0007       MAE: 0.0191
Normalized (Test):    MSE: 0.0008       MAE: 0.0190
Norm (Test 2025):     MSE: 0.0175       MAE: 0.1088 --> Very high
Real scale (Train):   RMSE: 104.7396$   MAE: 77.2301$   MAPE: 3.6412%
Real scale (Test):    RMSE: 114.3909$   MAE: 76.6764$   MAPE: 3.8678%
Real (Test 2025):     RMSE: 534.6767$   MAE: 439.3730$   MAPE: 14.2091%
"""

# ---------------------------------------------------------------------
#                       SOME TO-DOs and notes
# To add new ETH prices, we need to preserve X_min and X_max (normParams located at the models' files)

# Data 2017-2024 Starting MAPE: 400% Why???
# Graph errors and predictions without overloading the graph with data (not responding)
# Add every error (include 2025 Test) to the graph
# Add new artificial data
# Search for data leaking
# Add ALL the data the model needs to predict (includin in 2025)

print("-" * 10)
# print("- Poder re-entrenar -")
# print("Datos de varios dias")
# print("Probar diferente arquitectura y LR (varios a la vez)")
print("Añadir Close-n con la media o Close actual en vez de cero")
print("Visualizar error con gráficas")
# print("Probar otros datos. Cuales?")
# print("-- Comparar predictions con el precio 2025 --")
print(" HAY OVERFITTING? -> Test loss siempre baja. Raro // (Quitar precios 2021?)")
# print("-- Comprobar que no hay data leakage --")
# print("=== GUARDAR EN GITHUB === ")
print("Average price variation 2025: 85$ (Min: 0.2$, Max: 600$)")
print("-" * 10, "\n")


# Ver notas computación para mejorar la función de activacion
# [Unir y unificar datos en un solo archivo]
# Export model as JSON, not .txt (even to re-download it later?)
# Guardar modelos en .csv (JSON)
# Not currently storing "Date" on 'data' list
# REFACTOR: all_headers vs. model_headers

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
    print("Testing several models!")
    # --- TRAIN NEW MODELS ---
    X_train, X_test, Y_train, Y_test = utils.loadData(DATA_SPLIT)
    X_train_norm, Y_train_norm, norm_price = utils.normalizeTrainData(X_train, Y_train)
    X_test_norm, Y_test_norm = utils.normalizeTestData(X_test, Y_test, norm_price)
    X_2025, Y_2025 = utils.loadData()
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
    layers7 = [X_train.shape[0], 64, 64, 64, 1]  # Mejor, muy lento
    layers8 = [X_train.shape[0], 64, 64, 1]  # Regular-Bueno
    # layers9 = [X_train.shape[0], 64, 32, 16, 1] # MAL
    # layers_arr = [layers0, layers1, layers2, layers3, layers4, layers5, layers6]
    layers_arr = [layers2, layers3, layers7, layers8]
    # learning_rate = [0.001, 0.0001, 0.0008, 0.01, 0.03, 0.005]
    learning_rate = [0.01, 0.03, 0.005]
    models_arr = []
    print(f"Training {len(layers_arr)} models with {len(learning_rate)} different learning rates!")
    print(layers_arr, learning_rate)
    # TRAINING 1000 steps, learning rate = 0.0008, layers = [9, 64, 64, 32, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 32, 1] MUY BUENO (MAPE: 4-6)
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 32, 1] BUENO, UN POCO DE OVERFIT

    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 64, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 64, 1] MUY BUENO, UN POCO DE OVERFIT
    #
    # Close 1-5
    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 32, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 32, 1] BUENO
    # TRAINING: 1000 steps, learning rate = 0.005, layers = [9, 64, 64, 32, 1] BUENO

    # TRAINING: 1000 steps, learning rate = 0.001, layers = [9, 64, 64, 64, 1] MALO  6-8 mins
    # TRAINING: 1000 steps, learning rate = 0.0001, layers = [9, 64, 64, 64, 1] MALO
    # TRAINING: 1000 steps, learning rate = 0.0008, layers = [9, 64, 64, 64, 1] MALO
    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 64, 1] BUENO!?
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 64, 1] BUENO!?
    # TRAINING: 1000 steps, learning rate = 0.005, layers = [9, 64, 64, 64, 1] MED-BUENO

    # TRAINING: 1000 steps, learning rate = 0.001, layers = [9, 64, 64, 1] MALO 5 mins
    # TRAINING: 1000 steps, learning rate = 0.0001, layers = [9, 64, 64, 1] MALO
    # TRAINING: 1000 steps, learning rate = 0.0008, layers = [9, 64, 64, 1] MALO
    # TRAINING: 1000 steps, learning rate = 0.01, layers = [9, 64, 64, 1] MED-BUENO
    # TRAINING: 1000 steps, learning rate = 0.03, layers = [9, 64, 64, 1] MED-BUENO
    # TRAINING: 1000 steps, learning rate = 0.005, layers = [9, 64, 64, 1] BUENO
    """
    ERRORS (Test data):                       mse_norm_train, mse_norm_test, mae_norm_train, mae_norm_test, mape_train, mape_test
0 [9, 64, 64, 32, 1] 0.001 4.58 mins          [0.06823, 0.06341, 0.22167, 0.21295, 342.53465, 293.36657]
1 [9, 64, 64, 32, 1] 0.0001 5.42 mins         [0.10537, 0.10063, 0.28179, 0.27464, 401.01674, 344.42084]
2 [9, 64, 64, 32, 1] 0.0008 4.43 mins         [0.07726, 0.07257, 0.2392, 0.2317, 358.2586, 307.63269]
3 [9, 64, 64, 32, 1] 0.01 4.43 mins           [0.00206, 0.00191, 0.03901, 0.03764, 44.88268, 37.35677]
4 [9, 64, 64, 32, 1] 0.03 5.5 mins            [0.00089, 0.00089, 0.02551, 0.02579, 23.26977, 19.90166]
5 [9, 64, 64, 32, 1] 0.005 5.61 mins          [0.00807, 0.00754, 0.07024, 0.06746, 121.74889, 102.28052]
6 [9, 64, 64, 64, 1] 0.001 7.89 mins          [0.06339, 0.05921, 0.21516, 0.20793, 328.89867, 281.77319]
7 [9, 64, 64, 64, 1] 0.0001 6.91 mins         [0.09668, 0.0904, 0.26613, 0.25628, 403.68313, 347.13428]
8 [9, 64, 64, 64, 1] 0.0008 7.99 mins         [0.05794, 0.05486, 0.20796, 0.20276, 310.09326, 265.9037]
9 [9, 64, 64, 64, 1] 0.01 6.94 mins           [0.00123, 0.00111, 0.03028, 0.02902, 37.61491, 31.13318]
10 [9, 64, 64, 64, 1] 0.03 7.39 mins          [0.0009, 0.00088, 0.02589, 0.02596, 24.41277, 20.82219]
11 [9, 64, 64, 64, 1] 0.005 6.9 mins          [0.00705, 0.0063, 0.06327, 0.05929, 116.49113, 97.32221]
12 [9, 64, 64, 1] 0.001 4.65 mins             [0.06263, 0.05853, 0.214, 0.20701, 326.96392, 280.13595]
13 [9, 64, 64, 1] 0.0001 4.78 mins            [0.10855, 0.10239, 0.28873, 0.27826, 416.59694, 359.7833]
14 [9, 64, 64, 1] 0.0008 4.78 mins            [0.05818, 0.05411, 0.20617, 0.19801, 312.8121, 267.67062]
15 [9, 64, 64, 1] 0.01 4.75 mins              [0.00256, 0.00269, 0.04014, 0.03901, 60.21349, 49.90229]
16 [9, 64, 64, 1] 0.03 5.03 mins              [0.00153, 0.00148, 0.03438, 0.03406, 33.6356, 28.57743]
17 [9, 64, 64, 1] 0.005 4.69 mins             [0.00896, 0.00803, 0.07384, 0.06949, 129.99985, 109.05127]
MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)
    """

    data_num = X_train.shape[1]
    # Training all models, one by one
    modelCount = 0
    for i in range(len(layers_arr)):
        for lr in learning_rate:
            model = NeuralNetwork(layers_arr[i], data_num, lr)
            print(" "*10, f"\n======== MODEL {modelCount}/{len(learning_rate)*len(layers_arr)} ========")
            modelCount += 1
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
                printErrors=2,
                showGraph=False
            )

            print("           - - - FINAL TEST: MODEL #{modelCount} - - -")
            print(
                f"TRAINING: {training_steps} steps, learning rate = {lr}, layers = {layers_arr[i]}"
            )
            errors = model.test(
                X_train_norm, Y_train_norm, X_test_norm, Y_test_norm, X_2025_norm, Y_2025_norm, norm_price, showGraph=False
            )
            errors = [round(float(err), 5) for err in errors]
            t_f = time.time()
            runningTime = round((t_f - t0) / 60, 2)
            print(f"Training took {runningTime} minutes!")
            models_arr.append({"model": model, "errors": errors, "t": runningTime, "lr": lr})

    print("=" * 15, "END", "=" * 15)

    # Prints all models
    print(
        f"\nERRORS (Test data):{" " * 23}mse_norm_train, mse_norm_test, mae_norm_train, mae_norm_test, mape_train, mape_test"
    )
    for i in range(len(models_arr)):
        layers = models_arr[i]["model"].layers
        formatting = " " * 3 * int(6 - len(layers))
        print(i, layers, models_arr[i]["lr"], f"{models_arr[i]["t"]} mins", formatting, models_arr[i]["errors"])
    # --- Option to save the model ---
    print("MSE < 0.01        MAE < 0.03       MAPE < 10%    (1-5%; 85$)")
    index = int(input("Choose a model to save: "))
    save = input(f"\nDo you want to save the model {index}? (Y/n) ")
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
    # model_name = "model_MAPE__3_4_17.txt"
    # model_name = "trained_w_2025_MAPE_3_14.txt"
    model_name = "model_v2.0.txt"

    user_input = input(f"Do you want to load {model_name}? (Y/n) ").lower()
    # print("USER_INPUT CHANGED!!!")
    # user_input = "n"
    # Whether to create a new model or test a saved one
    if user_input == "n":
        # --- TRAIN NEW MODEL ---
        # --- Loading and parsing data ---
        # print("-- Loading and parsing data --")
        X_train, X_test, Y_train, Y_test = utils.loadData(DATA_SPLIT)

        # --- Normalizing train data ---
        # Returning the min and max prices to reverse normalisation in MAPE
        X_train_norm, Y_train_norm, norm_price = utils.normalizeTrainData(
            X_train, Y_train
        )
        X_test_norm, Y_test_norm = utils.normalizeTestData(X_test, Y_test, norm_price)

        # --- Loading train data (2025 prices) ---
        # X_2025, Y_2025 = utils.loadRealData("data/ETH_1D_CMC OLD.csv")
        X_2025, Y_2025 = utils.loadData()
        X_2025_norm, Y_2025_norm = utils.normalizeTestData(X_2025, Y_2025, norm_price)

        # --- Creating the model ---
        layers = [X_train.shape[0], 64, 64, 32, 1]

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
            global graph
            graph.resetGraph()
            model.train(
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
        X_2025, Y_2025 = utils.loadData()
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
