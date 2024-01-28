## MLFLow Server

mlflow server --host 127.0.0.1 --port 8080

## Testowanie modelu

mlflow models serve -m /.../mlartifacts/273362149068262204/0ed3751c4dc84b55ab49d91e9d272759/artifacts/iris_model --no-conda -p 1234

curl http://127.0.0.1:1234/invocations -H 'Content-Type:
application/json' -d '{         "inputs": [[1, 2, 3, 4]]
}'

UWAGA: Jeśli w MLFLow widzimy nazwy kolumn w tabeli input, musimy podać takie same nazwy w poleceniu curl!

curl http://127.0.0.1:1234/invocations -H 'Content-Type: application/json' -d '{
    "inputs": [{"A": 1, "B": 2, "C": 3, "D": 4}]
}'


## Budowanie dockera

mlflow models build-docker \
  -m /.../mlartifacts/273362149068262204/0ed3751c4dc84b55ab49d91e9d272759/artifacts/iris_model \
  -n model-from-mlflow \
  --enable-mlserver
