import mlflow

mlflow.set_tracking_uri("http://localhost:8080")

from sklearn import datasets
from sklearn.model_selection import train_test_split

model = mlflow.sklearn.load_model(model_uri = "models:/iris_lr/1")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_val)

print(y_pred)