# Start CORS server
pip install -e .
mlflow server --host 127.0.0.1 --port 8080 --app-name mlflow_cors

# Build docker images

```
mlflow models build-docker --model-uri "runs:/5480bb939dfe47f8890bd7e34cc6a351/rf_apples" --name "rf_apples"


mlflow models generate-dockerfile -m models:/rf_apples_01/1 --enable-mlserver

docker build --tag 'a2nfinance/regression:1.0.0' . --network=host
```