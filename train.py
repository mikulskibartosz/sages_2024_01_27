import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump
import dvc.api


if __name__ == "__main__":
    params = dvc.api.params_show()
    print(params)

    train_data_file = sys.argv[1]
    val_data_file = sys.argv[2]
    model_file = sys.argv[3]

    train_data = np.loadtxt(train_data_file, delimiter=",")
    val_data = np.loadtxt(val_data_file, delimiter=",")

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_val = val_data[:, :-1]
    y_val = val_data[:, -1]

    params = {
        "solver": params["train"]["solver"],
        "penalty": params["train"]["penalty"],
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    print(X_train.shape, y_train.shape)
    print(X_train[:5])
    print(y_train[:5])

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)


    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    dump(model, model_file)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

