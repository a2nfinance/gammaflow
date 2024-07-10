import mlflow

model = mlflow.pyfunc.load_model("runs:/d0090d70a7df48409691a1c010ab13af/rf_apples")
model.serve(port=5000)
