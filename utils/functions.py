import numpy as np
import pandas as pd
import random

# Metadata: Date
# Target: Close
good_headers = ["Date", "Open", "High", "Low", "Close", "Volume"]


def loadRealData(file):
    raw = pd.read_csv(file, sep=";", header=0)
    data = rawToData(raw)
    print("Real data length:", len(data))
    return parseRealData(data)


def parseRealData(data):
    X = np.zeros((len(data), 5))
    Y = np.zeros(
        len(data),
    )
    for i in range(len(data)):
        count = 0
        for head in good_headers:
            elt = data[i][head]
            if head != "Date":
                # Close Price
                if head != "Close":
                    X[i][count] = elt
                    count += 1
                else:
                    Y[i] = elt

    return X.T, Y.T


def loadData(DATA_SPLIT):
    # raw = np.loadtxt(file)
    # print(f"Raw data: {raw}")
    # files = ["data/ETH_Day_2017-2022.csv",
    files = ["data/ETH_Day_2024-2025.csv"]
    data = []
    for file in files:
        raw = pd.read_csv(file, sep=",", header=0)
        data.extend(rawToData(raw))
    # Data matrix with shape (total_variables, data_length)
    # Contains metas, which aren't added to the Xs
    # for i in range(len(raw)):
    #     data.append(np.array(raw.iloc[i]))
    # Randomize data order
    random.shuffle(data)
    print("Data length:", len(data))
    return parseData(data, DATA_SPLIT)


# Loading params from text file (weights, biases, layers, norm_params...)
def loadModel(name):
    data = []
    with open(name, "r") as f:
        for line in f:
            data.append(line.strip().split(","))

    # print("Loading model:\n")
    # Weights and biases
    B_index = data.index(["Biases"])
    W = []
    for i in range(B_index - 1):
        line = data[i]
        # Filtering for info lines
        if len(line) == 1:
            line = line[0]
            if "W" in line and "Weights" not in line:
                shape = [line[3:6], line[8:11]]
                shape = (int(shape[0]), int(shape[1]))
                # print("shape", shape)
                matrix = []
                # Getting values from the next line to the whole matrix
                i += 1
                for j in range(i, i + shape[0]):
                    # print(i, j, data[j][:5])
                    # Converting to floats
                    data_num = []
                    for elt in data[j]:
                        data_num.append(float(elt))
                    matrix.append(data_num)
                matrix = np.reshape(matrix, shape)
                W.append(matrix)
    B = []
    l_index = data.index(["Layers"])
    for i in range(B_index, l_index):
        line = data[i]
        # (i, "line", line[:3])
        # Filtering for info lines
        if len(line) == 1:
            line = line[0]
            if "B" in line and "Biases" not in line:
                shape = [line[3:6], line[8:11]]
                shape = (int(shape[0]), int(shape[1]))
                # print("shape", shape)
                matrix = []
                # Getting values from the next line to the whole matrix
                i += 1
                for j in range(i, i + shape[0]):
                    # Converting to floats
                    data_num = []
                    for string in data[j]:
                        data_num.append(float(string))
                    matrix.append(data_num)
                matrix = np.reshape(matrix, shape)
                B.append(matrix)
    # Layers
    layers = []
    next_index = data.index(["X_mins"])
    for i in range(j + 1, next_index):
        if len(data[i]) >= 2:
            for layer in data[i]:
                layers.append(int(layer))
    num_index = data.index(["Data number"])
    num = int(data[num_index + 1][0])
    # Normalising parameters
    X_mins, X_maxs = [], []
    for elt in data[next_index + 1]:
        X_mins.append(float(elt))
    for elt in data[next_index + 3]:
        X_maxs.append(float(elt))
    norm_price = {
        "X_mins": X_mins,
        "X_maxs": X_maxs,
        "Y_min": float(data[next_index + 5][0]),
        "Y_max": float(data[next_index + 7][0]),
    }
    params = {"W": W, "B": B, "layers": layers, "num": num, "norm_price": norm_price}
    # print("\nLoaded model params:", params)
    return params


def rawToData(raw):
    raw_dicts = raw.to_dict(orient="records")
    headers = raw_dicts[0].keys()

    data = []
    for dict in raw_dicts:
        clean_dict = {}
        for head in good_headers:
            # Changing volume values from "50K" to 50_000
            val = dict[head]
            if type(dict[head]) == str:
                # Deleting thousands comma
                val = val.replace(",", "")
                if head == "Volume":
                    # Deleting decimal point
                    val = dict[head].replace(".", "")
                    if "K" in val:
                        new_volume = float(val[:-1]) * 1000
                    elif "M" in val:
                        new_volume = float(val[:-1]) * 1000000
                    else:
                        new_volume = float(val[:-1])
                    # Multiplying volume (parsed as float) by ETH price
                    val = new_volume * float(dict["Close"].replace(",", ""))
            clean_dict[head] = val
        data.append(clean_dict)
    return data


def parseData(data, DATA_SPLIT):
    # Separating in Xs and Ys
    # print("\nPARSING")
    X = np.zeros((len(data), 5))
    Y = np.zeros(
        len(data),
    )
    for i in range(len(data)):
        count = 0
        for head in good_headers:
            elt = data[i][head]
            # Not adding Date and Symbol
            if head != "Date":
                # Close Price
                if head != "Close":
                    X[i][count] = elt
                    count += 1
                else:
                    Y[i] = elt

    print("X", X.shape, "Y", Y.shape)

    # Splitting the data, dividing by percentage
    training = int(len(data) * DATA_SPLIT / 100)

    X_train = X[:training].T
    X_test = X[training:].T
    Y_train = Y[:training].T
    Y_test = Y[training:].T

    return X_train, X_test, Y_train, Y_test


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
    X_maxs = []

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
