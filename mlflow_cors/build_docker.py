import subprocess
import os

logged_model = "models:/rf_apples_01/1"
model_dir = "rf_apples_01_docker"
docker_file = model_dir + "/Dockerfile"
image_name = "a2nfinance/rf_apples:1.0.0"
def single_execute(command):
    command = command.decode("utf-8")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, pipesize=2048)
    for c in iter(lambda: process.stdout.read(1), b""):
        yield c
def manage_command(command):
    # command = command.decode("utf-8")
    result = os.popen(command).read()
    return result

manage_command(f"mlflow models generate-dockerfile --model-uri {logged_model} --output-directory {model_dir}")
    
with open(docker_file, 'r') as f:
    lines = f.readlines()
    for ind, line in enumerate(lines):
        if line.startswith('RUN apt-get -y update && apt-get install -y --no-install-recommends '):
            lines[ind] = line.split('\n')[0] + ' -y gcc g++\n'
    with open(docker_file, 'w') as f:
        f.writelines(lines)
    
manage_command(f"docker build -t {image_name} {model_dir}")