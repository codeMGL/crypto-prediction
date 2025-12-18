import numpy as np
import pandas as pd
import random

# Metadata: Date
# Target: Close
# CloseN refers to the closing price if the N previous days
model_headers = {
    # "X": ["Open", "High", "Low", "Volume", "Close1", "Close2", "Close3", "Close4", "Close5"],
    "X": ["Open", "High", "Low", "Volume", "Close1"],
    "Y": ["Close"],
}
# Good headers to take info from
good_headers = ["Date", "Open", "High", "Low", "Close", "Volume"]


def reverseCSV(file):
    # Read the CSV file
    data = pd.read_csv(file, sep=",", header=0)
    print("REVERSING")
    print(data.head())

    # Reverse the data
    reversed_data = data.iloc[::-1]

    # Save the reversed data to a new file
    reversed_data.to_csv("reversed_data.csv", index=False)
    print("Reversed data saved to 'reversed_data.csv'")
    print("-- END --")


# -- Loading and parsing data for training --
def loadData(DATA_SPLIT=False):
    ALL_FILES = [
        "data/ETH_Day_2017-2022.csv",
        "data/ETH_Day_2017-2024.csv",
        "data/ETH_Day_2021-2024.csv", #
        "data/ETH_Day_end 2024.csv", #
        "data/ETH_Day_2024-2025.csv",
    ]
    print("--- LOADING ALL DATA FROM 2017 TO 2025 ---")
    files = [
        "data/ETH_Day_2017-2024.csv",
        "data/ETH_Day_end 2024.csv",
        "data/ETH_1D_CMC.csv",
    ]
    # files = [ALL_FILES[2], ALL_FILES[3]]
    # files = [ALL_FILES[1], ALL_FILES[3]] # ERROR/DEBUG: Initial MAPE > 400%
    # files = [ALL_FILES[2], ALL_FILES[4]]
    if not DATA_SPLIT:
        # Just loading 2025 data
        files = ["data/ETH_1D_CMC.csv"]

    data = []
    for file in files:
        raw = pd.read_csv(file, sep=",", header=0)
        data.extend(rawToData(raw))
        
        # print("\n", "="*15, file)
        # print(rawToData(raw)[:5])
    # data: matrix with shape (total_variables, data_length)
    # Contains metas, which aren't added to the Xs
    # Randomize data order
    random.shuffle(data)
    print("Data length:", len(data))
    return parseData(data, DATA_SPLIT)


# Transforms data file into a clean dictionary including all Xs and Ys used by the model
def rawToData(raw):
    raw_dicts = raw.to_dict(orient="records")
    headers = raw_dicts[0].keys()
    # print(f"First date: {raw_dicts[0]["Date"]} Second date: {raw_dicts[1]["Date"]}")
    # print("!!! Second date MUST be older than first date !!!")

    data = []
    for i in range(len(raw_dicts)):
        # Row (Dict) contaning the day data of ETH
        day_data = raw_dicts[i]
        clean_dict = {}
        all_headers= model_headers["X"].copy()
        all_headers.extend(model_headers["Y"])
        all_headers.append("Date")
        for head in all_headers:
            # Adding headers that are part of the data file headers
            if head in headers:
                # Adding data from current day
                val = day_data[head]
                if isinstance(day_data[head], str):
                    # Deleting thousands comma
                    val = val.replace(",", "")
                    # Changing volume values from "50K" to 50_000
                    if head == "Volume":
                        # Deleting decimal point
                        val = day_data[head].replace(".", "")
                        if "K" in val:
                            new_volume = float(val[:-1]) * 1e3
                        elif "M" in val:
                            new_volume = float(val[:-1]) * 1e6
                        else:
                            new_volume = float(val[:-1])
                        # Multiplying volume (parsed as float) by ETH price
                        val = new_volume * float(day_data["Close"].replace(",", ""))
                # Adding the parsed data to the clean dict
                if head == "Date":
                    clean_dict[head] = val
                else:
                    clean_dict[head] = float(val)
            else:
                # Adding close prices for previous days (lower indices)
                if head[:5] == "Close" and len(head) > 5:
                    past_index = int(head[5])
                    if i - past_index >= 0:
                        # Adds data as a float
                        val = raw_dicts[i - past_index]["Close"]
                        if isinstance(val, str):
                            val = val.replace(",", "")
                            val = float(val)
                    else:
                        # There are no previous values available. DEBUG: Currently using 'Close' value
                        val = str(raw_dicts[i]["Close"]) # 0.0
                        val = float(val.replace(",", ""))
                    # Adding the parsed data to the clean dict
                    clean_dict[head] = val

        data.append(clean_dict)
    return data


def parseData(data, DATA_SPLIT):
    # Separating in Xs and Ys
    # print("\n-- PARSING DATA --")
    X = np.zeros((len(data), len(model_headers["X"])))
    Y = np.zeros((len(data), len(model_headers["Y"])))
    # print("Shapes", X.shape, Y.shape)
    # Data is a list of dictionaries containg every row of prices
    for i in range(len(data)):
        count = 0
        # Not adding Date and Symbol (they're not in model_headers)
        row = []
        all_headers = model_headers["X"].copy()
        all_headers.extend(model_headers["Y"])
        for head in all_headers:
            # Not adding metadata
            if head != "Date":
                # Separating the target variable (Close)
                if head != "Close":
                    # Adding 'head' data (into a list) to the Xs
                    row.append(data[i][head])
                else:
                    # Adding target value ("Close") to the Ys
                    elt = data[i][head]
                    Y[i] = elt
        # print("i", i, "row", row)
        X[i] = row
        count += 1

    print("\nAll train Xs", X.shape, "All train Ys", Y.shape)
    # print("---- DATA ----")
    # print(X[:2])
    # print(X.T[:2])
    # print("Y")
    # print(Y[:2])
    # print(Y.T[:2])

    if DATA_SPLIT:
        # Splitting the data, dividing by percentage
        print("error", len(data))
        print(DATA_SPLIT)
        training = int(len(data) * DATA_SPLIT / 100)

        X_train = X[:training].T
        X_test = X[training:].T
        Y_train = Y[:training].T
        Y_test = Y[training:].T

        return X_train, X_test, Y_train, Y_test
    else:
        # Returning 100% of the 2025 data
        return X.T, Y.T


# -- Loading and parsing real data for validation --
def loadRealData(file):
    raw = pd.read_csv(file, sep=";", header=0)
    data = rawToData(raw)
    print("Real data length:", len(data))
    return parseRealData(data)


def parseRealData(data):
    X = np.zeros((len(data), len(model_headers["X"])))
    Y = np.zeros(
       ( len(data), len(model_headers["Y"]))
    )

    # REFACTOR: Probably can be condensed and optimised with array functions + numpy
    for i in range(len(data)):
        count = 0
        for head in good_headers:
            if head != "Date":
                # print(i, "data", data[i])
                elt = data[i][head]
                # Close Price
                if head != "Close":
                    X[i][count] = elt
                    count += 1
                else:
                    Y[i] = elt

    # --- Getting price difference between 2 consecutive days in the ETH_1D_CMC (2025) file ---
    # price_diference, sum = [], 0
    # for i in range(len(X)):
    #     # Sanitazing output for 1 target. REFACTOR/DEBUG
    #     Y_i = Y[i]
    #     if isinstance(Y[i], np.ndarray) and len(Y[i]) == 1:
    #         Y_i = Y[i][0]
    #     diff = Y_i - X[i][0]
    #     # print("Open", X[i][0], "Close", Y_i, "diff", diff)
    #     price_diference.append(round(diff, 3))
    #     sum += diff
    # avg = np.round(sum / len(X), 5)
    # print("\n-- Price difference between open and close (2025) --")
    # # print("Price diff (close - open):", price_diference[:20])
    # print(f"Max: {max(price_diference)}$  Min: {min(price_diference)}$  Average: {avg}$")
    # # Absolute value
    # sum = 0
    # for i in range(len(price_diference)):
    #     price_diference[i] = abs(price_diference[i])
    #     sum += price_diference[i]
    # avg = np.round(sum / len(X), 5)
    # # print("\nPrice diff (Absolute values):", price_diference[:20])
    # print("-- Price difference (Absolute values) --")
    # print(f"Max: {max(price_diference)}$  Min: {min(price_diference)}$  Average: {avg}$")

    # print("(2025 data) X", X.shape, "Y", Y.shape)
    # print("---- 2025 DATA ----")
    # print(X[:2])
    # print(X.T[:2])
    # print("Y")
    # print(Y[:2])
    # print(Y.T[:2])

    return X.T, Y.T


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
    print("X\n", X.shape, "\n", X[:2], X.T[:2])
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
    print("norm_params:\n", norm_params)

    return X_norm, Y_norm, norm_params


def normalizeTestData(X, Y, norm_params):
    # Normaliza X e Y a rango [0, 1]
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
    y_min, y_max = norm_params["Y_min"], norm_params["Y_max"]
    # for i in range(Y_norm.shape[0]):
    #     Y[i] = Y_norm[i] * (max - min) + min
    Y = Y_norm * (y_max - y_min) + y_min
    return Y
