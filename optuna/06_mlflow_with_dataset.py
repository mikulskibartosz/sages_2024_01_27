import mlflow

mlflow.set_tracking_uri("http://localhost:8080")

from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("IRIS LR with Data")
mlflow.autolog()

with mlflow.start_run():
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
        "multi_class": "auto",
    }

    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    signature = infer_signature(X_train, lr.predict(X_train))

    print(accuracy)