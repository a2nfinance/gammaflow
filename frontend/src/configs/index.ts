export const MLFLOW_SERVER_API = `${process.env.NEXT_PUBLIC_MLFLOW_TRACKING_SERVER}/api`;
export const CREATE_EXPERIMENT_ENDPOINT = `${MLFLOW_SERVER_API}/2.0/mlflow/experiments/create`
export const GET_EXPERIMENT_ENDPOINT = `${MLFLOW_SERVER_API}/2.0/mlflow/experiments/get`
export const SEARCH_EXPERIMENT_ENDPOINT = `${MLFLOW_SERVER_API}/2.0/mlflow/experiments/search`
export const SEARCH_RUNS= `${MLFLOW_SERVER_API}/2.0/mlflow/runs/search`
export const GET_RUN = `${MLFLOW_SERVER_API}/2.0/mlflow/runs/get`
export const GET_ARTIFACTS_LIST= `${MLFLOW_SERVER_API}/2.0/mlflow/artifacts/list`
export const NETWORK_EXPLORER = `https://explorer.thetatoken.org/`


export const SEARCH_MODEL_ENDPOINT= `${MLFLOW_SERVER_API}/2.0/mlflow/model-versions/search`
export const CREATE_REGISTERED_MODEL_ENDPOINT = `${MLFLOW_SERVER_API}/2.0/mlflow/registered-models/create`
export const CREATE_MODEL_VERSION_ENDPOINT = `${MLFLOW_SERVER_API}/2.0/mlflow/model-versions/create`
export const GET_DOWNLOAD_URI_FOR_MODEL_VERSION_ARTIFACTS_ENDPOINT= `${MLFLOW_SERVER_API}/2.0/mlflow/model-versions/get-download-uri`

