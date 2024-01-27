import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


DATA = "data/iris.csv"


os.makedirs("model", exist_ok=True)
df = pd.read_csv(DATA)
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
}

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtest, "test")],
    early_stopping_rounds=10,
)

model.save_model("model/model.xgb")
model.dump_model("model/model.txt")