## Zdalne repozytorium w Google Drive

dvc remote add google gdrive://.../dvcstore

dvc remote modify google gdrive_acknowledge_abuse true

dvc push -r google / dvc pull -r google

## DVC Pipeline

### Dodawanie kroków

dvc stage add -n preprocess -d preprocess.py -o data/preprocessed/iris_train.csv -o data/preprocessed/iris_validate.csv python3 preprocess.py data/preprocessed/iris_train.csv data/preprocessed/iris_validate.csv

dvc stage add -n train -d train.py -d data/preprocessed/iris_train.csv -d data/preprocessed/iris_validate.csv -o model/model.pkl python3 train.py data/preprocessed/iris_train.csv data/preprocessed/iris_validate.csv model/model.pkl

dvc stage add -n eval -d eval.py -d model/model.pkl -d data/preprocessed/iris_validate.csv python3 eval.py model/model.pkl data/preprocessed/iris_validate.csv

### Edytowanie kroków

dvc stage add --force -n train -p train.solver,train.penalty -d train.py -d data/preprocessed/iris_train.csv -d data/preprocessed/iris_validate.csv -o model/model.pkl python3 train.py data/preprocessed/iris_train.csv data/preprocessed/iris_validate.csv model/model.pkl

### Uruchomienie kroków (kroki już wykonane będą pominięte)

dvc repro

### Wyświetlenie grafu zależności

dvc dag

### Uruchomienie eksperymentu

dvc exp run -n train

#### Uruchomienie z parametrami

Aby dodać nowe parametry:

dvc exp run -n train -S +train.solver=lbfgs -S +train.penalty=l2

Każde kolejne uruchomienie:

dvc exp run -n train -S train.solver=lbfgs -S train.penalty=l2

Pominięte parametry są czytane z pliku params.yaml (mogą pochodzić z poprzedniego eksperymentu)

### Wyświetlenie listy eksperymentów

dvc exp show