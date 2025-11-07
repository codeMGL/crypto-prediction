import numpy as np
import pandas as pd

from main import DATA_SPLIT

def loadData(file):
    # raw = np.loadtxt(file)
    # print(f"Raw data: {raw}")

    raw = pd.read_csv(file, sep=",", header=0)[:10]
    print(type(raw), "iloc\n", raw.iloc[:])
    # Data matrix with shape (total_variables, data_length)
    # Contains metas, which aren't added to the Xs
    data = []
    for i in range(len(raw)):
        data.append(np.array(raw.iloc[i]))
    print("DATA:", len(data))
    return parseData(data)


def parseData(data):
    # Separating in Xs and Ys
    print("\nPARSING")
    X = np.zeros((len(data), 6))
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

    print("X", X.shape, "Y", Y.shape)

    # Splitting the data, dividing by percentage
    training= int(len(data) * DATA_SPLIT / 100)

    X_train = X[:training].T
    X_test = X[training:].T
    Y_train = Y[:training].T
    Y_test = Y[training:].T

    return [X_train, X_test, Y_train, Y_test]
