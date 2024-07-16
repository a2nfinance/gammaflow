# Create a background service
sudo vi /etc/systemd/system/mlflow-tracking.service
sudo systemctl daemon-reload
# Start CORS server
pip install -e .
mlflow server --host 127.0.0.1 --port 8080 --app-name mlflow_cors

# Build docker images

```
mlflow models build-docker --model-uri "runs:/949630bb34d54b55b227c652776d5e9d/rf_apples" --name "a2nfinance/rf_apples:1.0.0"
mlflow models build-docker --model-uri "runs:/d94ba393ba3d4de1831e1c0dcf8f1808/whisper_transcriber" --name "a2nfinance/audio_to_text:1.0.6"


mlflow models generate-dockerfile -m models:/whisper_transcriber/1 --enable-mlserver

docker build --tag 'a2nfinance/regression:1.0.0' . --network=host
```
# Run serve

mlflow models serve -m runs:/949630bb34d54b55b227c652776d5e9d/rf_apples -p 5000
mlflow models serve -m runs:/d94ba393ba3d4de1831e1c0dcf8f1808/whisper_transcriber -h 0.0.0.0 -p 5000
export MLFLOW_TRACKING_URI=http://34.16.145.233:8080

# Login 
docker login -u "levi@a2n.finance" -p "password" docker.io
mlflow models generate-dockerfile -m models:/whisper_transcriber/1 --output-directory audio_to_text_docker

## Data input
[{"average_temperature": 10,"rainfall": 20, "weekend": 100, "holiday": 10, "price_per_kg": 20, "promo": 20, "previous_days_demand": 100}]

curl http://localhost:5001/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"average_temperature": 10,"rainfall": 20, "weekend": 100, "holiday": 10, "price_per_kg": 20, "promo": 20, "previous_days_demand": 100}]}'
curl http://34.125.25.91:6060/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"audio": ["aaaa"]}]}'
curl http://34.125.25.91:6060/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"audio": ["ssss"]}]}'