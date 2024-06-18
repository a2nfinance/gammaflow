from models import Generator_I
import os
from trainer import loadState, save_video
import torch.nn as nn
import os
from glob import glob
from text_to_class.models import LSTM
from text_to_class.dataloading import TextLoader
import torch
import torch


gen = Generator_I(nc=3, ngf=64, nz=60, ngpu=1, nClasses= 102, batch_size= 16)

# Definde a state path
current_path = os.getcwd()
trained_path = os.path.join(current_path, 'trained_models')

# Load pre_trained states
loadState(9, gen, path = trained_path)

# Load LSTM model to get the category predicted from natural language
rnnType     = nn.LSTM
rnnSize     = 512
embedSize   = 256
itemLength  = 10
loadEpoch   = 1000

dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'processed_dataset.txt')
dataset_path = glob(dataset_path)

if not dataset_path:
    raise FileNotFoundError(f"No dataset found at {dataset_path}")

dataset_path = dataset_path[0] 
dataset = TextLoader(dataset_path, item_length = itemLength)
vocal_size = len(dataset.vocabulary) 

network = LSTM(rnnType, rnnSize, embedSize, vocal_size)
network.loadState(loadEpoch, os.path.join(current_path, 'text_to_class/'))

'''
1. Put a input and let's generate a video. 
2. Take your input and tell you the predicted class.
3. Generate and save the video.
'''

humanDescription     = input('Put your input here: > ')

try:
    toForwardDescription = dataset.prepareTxtForTensor(humanDescription)
    results              = network(torch.tensor(toForwardDescription).unsqueeze_(0))
    _, actionIDx         = results.max(1)
    actionClassName      = dataset.getClassNameFromIndex(actionIDx.item() + 1)
    print(f'Predicted class is {actionClassName}')    
    print(actionIDx)
except KeyError as err:
    print('Sorry, that word is not in the vocabulary. Please try again.')

mean   = (100.99800554447337/255, 96.7195209000943/255, 89.63882431650443/255)
std    = (72.07041943699456/255, 70.41506399740703/255, 71.55581999303428/255)
if torch.cuda.is_available():
    gen      = gen.cuda()

n_videos = 1
n_frames = 25 * 3 # 3s


n_channels      = 3
dim_z_category  = 60
dim_z_motion    = 10
video_length    = 16


save_path =  current_path
actionIDx       = torch.tensor(dim_z_category - 2) if actionIDx.item() >= dim_z_category else actionIDx

fakeVideo, _ = gen.sample_videos(n_videos, n_frames, [actionIDx.item()])
fakeVideo    = fakeVideo[0].detach().cpu().numpy().transpose(1, 2, 3, 0)
save_video(fakeVideo, actionClassName, 0, std, mean, save_path)