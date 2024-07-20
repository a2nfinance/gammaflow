##############################################################################################
## The source code is referenced from Text2VideoGAN                                         ##
## (see https://github.com/CarloP95/Text2VideoGAN).                                         ##
##############################################################################################

from generate_videos.models import VideoGenerator
import os
from generate_videos.trainer import loadState, save_video
import torch.nn as nn
import os
from glob import glob
from text_to_class.models import LSTM
from text_to_class.dataloading import TextLoader
import torch
import argparse


'''
1. Put a input and let's generate a video. 
2. Take your input and tell you the predicted class.
3. Generate and save the video.
'''
def text_to_video(text):
    
    parser = argparse.ArgumentParser(description='Start generating GammaFlow.....')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Set to use the GPU.')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='set the number of gpu you use')
    parser.add_argument('--video_path', type=str, default='generate_videos/trained_models/VideoGenerator_epoch-120000',
                        help='set path (prefix name) to load state for generating video')
    parser.add_argument('--text_path', type=str, default='text_to_class/LSTM-checkpoint-3700',
                        help='set path (prefix name) to load state for text to class')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples correspond to a batch size')
    parser.add_argument('--nClasses', type=int, default=11,
                        help='Number of classes on which the Embedding module will work.')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Parameter of the ConvTranspose2d Layers.')

    args       = parser.parse_args()
    cuda       = args.cuda
    ngpu       = args.ngpu
    video_path = args.video_path
    text_path  = args.text_path

    num_samples = args.num_samples
    nClasses = args.nClasses
    ngf = args.ngf

    gen = VideoGenerator(nc=3, ngf=ngf, nz = 60, ngpu=ngpu, nClasses= nClasses, batch_size= num_samples)

    # Definde a state path
    current_path = os.getcwd()
    trained_path = os.path.join(current_path, video_path)
    # Load pre_trained states
    loadState(gen, path = trained_path)

    # Load LSTM model to get the category predicted from natural language
    rnnType     = nn.LSTM
    rnnSize     = 512
    embedSize   = 512
    itemLength  = 30
    loadEpoch   = 3700

    dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'action_classes.txt')
    dataset_path = glob(dataset_path)

    if not dataset_path:
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    dataset_path = dataset_path[0] 
    dataset = TextLoader(dataset_path, item_length = itemLength)
    vocal_size = len(dataset.vocabulary) 

    network = LSTM(rnnType, rnnSize, embedSize, vocal_size, ngpu=ngpu)
    network.loadState(os.path.join(current_path, text_path))
    try:
        toForwardDescription = dataset.prepareTxtForTensor(text)
        results              = network(torch.tensor(toForwardDescription).unsqueeze_(0))
        _, actionIDx         = results.max(1)
        actionClassName      = dataset.getClassNameFromIndex(actionIDx.item() + 1)
        # print(f'Predicted class is {actionClassName}')    
        # print(actionIDx)
    except KeyError as err:
        print('Sorry, that word is not in the vocabulary. Please try again.')

    if torch.cuda.is_available():
        gen      = gen.cuda()

    video_len = 25*5
    save_path =  current_path + "/output_videos/"
    fakeVideo = gen.sample_videos(video_len, actionIDx.item() + 1)
    fakeVideo = fakeVideo[0].detach().cpu().numpy().transpose(1, 2, 3, 0)
    output_path = save_video(fakeVideo, actionClassName, save_path)
    return output_path


