import os
import torch
import torch.nn as nn
from glob import glob
from argparse import ArgumentParser
import mlflow
import mlflow.pytorch
from generate_videos.models import VideoGenerator
from generate_videos.trainer import loadState, save_video
from text_to_class.models import LSTM
from text_to_class.dataloading import TextLoader
from mlflow.models import infer_signature

# Function to initialize MLflow
def initialize_mlflow():
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_tracking_uri("http://34.125.25.91:8080")
    mlflow.set_experiment(experiment_id="560267043823286999")

# Function to parse arguments
def parse_arguments():
    parser = ArgumentParser(description='Start generating GammaFlow.....')
    parser.add_argument('--cuda', type=bool, default=False, help='Set to use the GPU.')
    parser.add_argument('--ngpu', type=int, default=1, help='Set the number of GPUs you use')
    parser.add_argument('--video_path', type=str, default='generate_videos/trained_models/VideoGenerator_epoch-120000',
                        help='Set path (prefix name) to load state for generating video')
    parser.add_argument('--text_path', type=str, default='text_to_class/LSTM-checkpoint-3700',
                        help='Set path (prefix name) to load state for text to class')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples corresponding to a batch size')
    parser.add_argument('--nClasses', type=int, default=11, help='Number of classes on which the Embedding module will work.')
    parser.add_argument('--ngf', type=int, default=64, help='Parameter of the ConvTranspose2d Layers.')
    return parser.parse_args()

# Function to load models
def load_models(args):
    gen = VideoGenerator(nc=3, ngf=args.ngf, nz=60, ngpu=args.ngpu, nClasses=args.nClasses, batch_size=args.num_samples)
    current_path = os.getcwd()
    trained_path = os.path.join(current_path, args.video_path)
    loadState(gen, path=trained_path)

    rnnType = nn.LSTM
    dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'action_classes.txt')
    dataset_path = glob(dataset_path)

    if not dataset_path:
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    dataset_path = dataset_path[0]
    dataset = TextLoader(dataset_path, item_length=30)
    vocal_size = len(dataset.vocabulary)

    network = LSTM(rnnType, 512, 512, vocal_size, ngpu=args.ngpu)
    network.loadState(os.path.join(current_path, args.text_path))

    return gen, network, dataset

# Function to generate video
def generate_video(gen, network, dataset, humanDescription):
    toForwardDescription = dataset.prepareTxtForTensor(humanDescription)
    results = network(torch.tensor(toForwardDescription).unsqueeze_(0))
    _, actionIDx = results.max(1)
    actionClassName = dataset.getClassNameFromIndex(actionIDx.item() + 1)

    if torch.cuda.is_available():
        gen = gen.cuda()

    video_len = 25 * 5
    save_path = os.path.join(os.getcwd(), "video_output")
    fakeVideo = gen.sample_videos(video_len, actionIDx.item() + 1)
    fakeVideo = fakeVideo[0].detach().cpu().numpy().transpose(1, 2, 3, 0)
    save_video(fakeVideo, actionClassName, save_path)
    
    return save_path, actionClassName

# Custom PyTorch model that combines the LSTM and VideoGenerator
class TextToVideoModel(nn.Module):
    def __init__(self, lstm_model, video_generator, dataset, ngpu=1):
        super(TextToVideoModel, self).__init__()
        self.lstm_model = lstm_model
        self.video_generator = video_generator
        self.dataset = dataset
        self.ngpu = ngpu

    def forward(self, text_input):
        toForwardDescription = self.dataset.prepareTxtForTensor(text_input)
        results = self.lstm_model(torch.tensor(toForwardDescription).unsqueeze_(0))
        _, actionIDx = results.max(1)

        if torch.cuda.is_available() and self.ngpu > 0:
            self.video_generator = self.video_generator.cuda()

        video_len = 25 * 5
        fakeVideo = self.video_generator.sample_videos(video_len, actionIDx.item() + 1)
        fakeVideo = fakeVideo[0].detach().cpu().numpy().transpose(1, 2, 3, 0)
        
        return fakeVideo

# Main function
def main():
    initialize_mlflow()
    args = parse_arguments()

    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))

        gen, network, dataset = load_models(args)

        humanDescription = input('Put your input here: > ')

        try:
            video_file_path, actionClassName = generate_video(gen, network, dataset, humanDescription)
            mlflow.log_param("action_class", actionClassName)

            if video_file_path:
                mlflow.log_artifact(video_file_path)
                print(f"Video saved to {video_file_path}")
            else:
                print("")
        except KeyError as err:
            print('Sorry, that word is not in the vocabulary. Please try again.')

        # Combine the models into a single pipeline model
        combined_model = TextToVideoModel(network, gen, dataset, ngpu=args.ngpu)
        
        # Define input example and signature
        text_input_example = "example text"
        video_output_example = combined_model(text_input_example)
        signature = infer_signature(text_input_example, video_output_example)

        # Save the combined pipeline model with signatures
        mlflow.pytorch.log_model(combined_model, "text_to_video_pipeline", signature=signature, input_example=text_input_example)

if __name__ == "__main__":
    main()