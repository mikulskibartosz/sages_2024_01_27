from joblib import load
from sklearn.metrics import accuracy_score
import sys
import numpy as np
from dvclive import Live

model_path = sys.argv[1]
val_data_file = sys.argv[2]

with Live() as live:
    model = load(model_path)

    val_data = np.loadtxt(val_data_file, delimiter=",")
    X_val = val_data[:, :-1]
    y_val = val_data[:, -1]

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    live.log_params({"whatever": 123})

    live.log_metric("accuracy", accuracy)

    live.log_artifact(model_path, type="model", name="iris_lr")

