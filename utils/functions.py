import numpy as np
import pandas as pd
import random


def loadData(DATA_SPLIT):
    # raw = np.loadtxt(file)
    # print(f"Raw data: {raw}")
    files = ["data/ETH_day.txt", "data/ETH_Day_2017-2022.csv"]
    for file in files:
        print(f"\nLoading: {file}")
        raw = pd.read_csv(file, sep=",", header=0)
        print(raw.iloc[:20])
        # Data matrix with shape (total_variables, data_length)
        # Contains metas, which aren't added to the Xs
        data = []
        for i in range(len(raw)):
            data.append(np.array(raw.iloc[i]))
    print("Data length:", len(data))
    # Randomize data order
    random.shuffle(data)
    return parseData(data, DATA_SPLIT)


def parseData(data, DATA_SPLIT):
    # Separating in Xs and Ys
    # print("\nPARSING")
    X = np.zeros((len(data), 5))
    Y = np.zeros(
        len(data),
    )
    for i in range(len(data)):
        count = 0
        for j in range(len(data[i])):
            # Not adding Date and Symbol
            if j != 0 and j != 1:
                # Close Price
                if j != 5:
                    X[i][count] = data[i][j]
                    count += 1
                else:
                    Y[i] = data[i][j]

    # print("X", X.shape, "Y", Y.shape)

    # Splitting the data, dividing by percentage
    training = int(len(data) * DATA_SPLIT / 100)

    X_train = X[:training].T
    X_test = X[training:].T
    Y_train = Y[:training].T
    Y_test = Y[training:].T

    return [X_train, X_test, Y_train, Y_test]


# def normalizeData(X, Y):
#     # Normalizing characteristics
#     # print("\nNormalizing:")
#     X_norm = X.copy()
#     Y_norm = Y.copy()
#     for i in range(X.shape[0]):
#         min, max = 10**5, 0
#         for j in range(X.shape[1]):
#             val = X[i][j]
#             if val > max:
#                 max = val
#             if val < min:
#                 min = val
#         for j in range(X.shape[1]):
#             X_norm[i][j] = (X[i][j] - min) / (max - min)

#     # Normalizing target variable
#     if len(Y.shape) > 1:
#         raise ValueError("Output has 2 targets, only normalizing one")
#     min, max = 10**5, 0
#     for i in range(Y.shape[0]):
#         val = Y[i]
#         if val > max:
#             max = val
#         if val < min:
#             min = val
#     for i in range(Y.shape[0]):
#         Y_norm[i] = (Y[i] - min) / (max - min)


#     return [X_norm, Y_norm, {"min": min, "max": max}]
# ERROR corregido de Claude


def normalizeTrainData(X, Y):
    """
    Normaliza X e Y a rango [0, 1]
    """
    X_norm = X.copy()
    Y_norm = Y.copy()

    X_mins = []
    X_maxs=[]

    # Normalizar X (cada feature por separado)
    for i in range(X.shape[0]):
        min_val = np.min(X[i, :])
        max_val = np.max(X[i, :])

        # Guardamos los valores máximo y mínimo
        X_mins.append(min_val)
        X_maxs.append(max_val)

        X_norm[i, :] = (X[i, :] - min_val) / (max_val - min_val + 1e-10)

    # Normalizar Y (todo junto)
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-10)

    norm_params = {
        "X_mins": np.array(X_mins),
        "X_maxs": np.array(X_maxs),
        "Y_min": Y_min,
        "Y_max": Y_max,
    }

    return X_norm, Y_norm, norm_params


def normalizeTestData(X, Y, norm_params):
    """
    Normaliza X e Y a rango [0, 1]
    """
    X_norm = X.copy()
    Y_norm = Y.copy()

    # Normalizar X (cada feature por separado)
    min_vals = norm_params["X_mins"]
    max_vals = norm_params["X_maxs"]
    for i in range(X.shape[0]):
        X_norm[i, :] = (X[i, :] - min_vals[i]) / (max_vals[i] - min_vals[i] + 1e-10)

    # Normalizar Y (todo junto)
    Y_min = norm_params["Y_min"]
    Y_max = norm_params["Y_max"]
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-10)

    return X_norm, Y_norm


# Un-normalizing Y data
def unNormalizeData(Y_norm, norm_params):
    Y = Y_norm.copy()
    min, max = norm_params["Y_min"], norm_params["Y_max"]
    # for i in range(Y_norm.shape[0]):
    #     Y[i] = Y_norm[i] * (max - min) + min
    Y = Y_norm * (max - min) + min
    return Y
