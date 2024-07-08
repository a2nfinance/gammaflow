import sys
import os
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_videos.models import VideoGenerator, VideoDiscriminator
import os
from generate_videos.trainer import loadState, save_video
import os
from glob import glob
from text_to_class.dataloading import TextLoader
import torch
import argparse
from torch.autograd import Variable
from torch import nn

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

criterion = nn.BCELoss()

def find_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

def bp(inputs, y, dis, retain=False):
    #print("----BackPropagate_V-----")
    #print(inputs.size())
    if cuda :
        label = (torch.FloatTensor()).cuda()
    else:
        label = torch.FloatTensor()
    try:
        label.resize_(inputs.size(0)).fill_(y)

    except RuntimeError as _:
        # Dimension of y does not allow to use fill_
        assert(inputs.size(0) == y.size(0))
        if cuda:
            label = (torch.FloatTensor(y)).cuda()
        else:
            label = torch.FloatTensor(y)

    labelv = Variable(label)
    outputs, _ = dis(inputs)
   
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    toReturnErr = err.data[0] if err.size() == torch.Tensor().size() else err.item()
    #print("----End of BackPropagate_V-----")
    return toReturnErr, outputs.data.mean()

gen = VideoGenerator(nc=3, ngf=ngf, nz = 60, ngpu=ngpu, nClasses= nClasses, batch_size= num_samples)
dis = VideoDiscriminator(nc=3, ndf=64, T=16, ngpu=ngpu)

# Definde a state path
current_path = os.getcwd()
trained_path = os.path.join(current_path, video_path)

# Load pre_trained states
loadState(gen, path = trained_path)
loadState(dis, path = trained_path)

itemLength  = 30

dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'action_classes.txt')
dataset_path = glob(dataset_path)

if not dataset_path:
    raise FileNotFoundError(f"No dataset found at {dataset_path}")

filenameDictClassesIdx = "classInd.txt"
dictClassesIdx = {}

with open(os.path.join(current_path, "generate_videos", "classes", filenameDictClassesIdx)) as file:
    for line in file:
        dictClassesIdx[ line.split() [1]] = int( line.split() [0] )

dataset_path = dataset_path[0] 
dataset = TextLoader(dataset_path, item_length = itemLength)
# vocal_size = len(dataset.vocabulary) 


if torch.cuda.is_available():
    gen      = gen.cuda()

video_len = 16
save_path =  current_path + "/video_output/"

fakeVideo = gen.sample_videos(video_len, 1)
values = list(dictClassesIdx.values())

for v in values:
    err_G, _ = bp(fakeVideo, v/len(dictClassesIdx), dis, retain=True)
    print("Generate class %s with a generation loss: %.4f" %(find_key_by_value(dictClassesIdx, v), err_G))







