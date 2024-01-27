import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna


DATA = "data/iris.csv"


os.makedirs("model", exist_ok=True)
df = pd.read_csv(DATA)
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

def objective(trial):
    trial_params = params.copy()
    trial_params["max_depth"] = trial.suggest_int("max_depth", 1, 9, log=True)
    trial_params["gamma"] = trial.suggest_float("gamma", 0.01, 1.0, log=True)
    trial_params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    trial_params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0, log=True)

    model = xgb.train(
        params=trial_params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    y_pred = model.predict(dtest)
    acc = accuracy_score(y_test, y_pred)
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_trial = study.best_trial
print(f"Best trial (log loss): {best_trial.value}")
print("Best trial (params):")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


model = xgb.train(
    params=best_trial.params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtest, "test")],
    early_stopping_rounds=10,
)

model.save_model("model/model.xgb")
model.dump_model("model/model.txt")