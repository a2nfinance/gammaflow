import argparse
from text_to_class.dataloading import TextLoader
import torch
import torch.nn as nn
import os
from glob import glob
from ..text_to_class.models import LSTM

parser = argparse.ArgumentParser(description='Testing text to class.....')
parser.add_argument('--path', default = 'data/action_classes.txt', type = str,
                            help= 'Set the relative path to find the file that contains the dataset.')
parser.add_argument('--sequence_length', default = 30, type = int,
                            help= 'Set the maximum length for each item that will be given to the model.')
parser.add_argument('--text_path', type=str, default='text_to_class/LSTM-checkpoint-3700',
                     help='set path (prefix name) to load state for text to class')


args = parser.parse_args()
path = args.path
sequence_len = args.sequence_len
text_path = args.text_path

# Load LSTM model to get the category predicted from natural language
rnnType     = nn.LSTM
rnnSize     = 512
embedSize   = 512
itemLength  = 30
loadEpoch   = 3700

current_path = os.getcwd()
dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'action_classes.txt')
dataset_path = glob(dataset_path)

if not dataset_path:
    raise FileNotFoundError(f"No dataset found at {dataset_path}")

dataset_path = dataset_path[0] 
dataset = TextLoader(dataset_path, item_length = itemLength)
vocal_size = len(dataset.vocabulary) 

network = LSTM(rnnType, rnnSize, embedSize, vocal_size, ngpu=1)
network.loadState(os.path.join(current_path, text_path))

'''
1. Put a input and let's generate a video. 
2. Take your input and tell you the predicted class.
'''

humanDescription     = input('Put your input here: > ')

try:
    toForwardDescription = dataset.prepareTxtForTensor(humanDescription)
    results              = network(torch.tensor(toForwardDescription).unsqueeze_(0))
    _, actionIDx         = results.max(1)
    actionClassName      = dataset.getClassNameFromIndex(actionIDx.item() + 1)
    #print(f'Predicted class is {actionClassName}')    
    #print(actionIDx)
except KeyError as err:
    print('Sorry, that word is not in the vocabulary. Please try again.')


dataset         = TextLoader(path, item_length= sequence_len)

text                = input('Input a string to test how does the model performs >')
if torch.cuda.is_available():
    tensor              = torch.tensor(dataset.prepareTxtForTensor(text)).cuda().unsqueeze_(0)
else: 
    tensor              = torch.tensor(dataset.prepareTxtForTensor(text)).unsqueeze_(0)

output              = network(tensor)
probability, action = output.max(1)
print(f'Predicted class is {dataset.getClassNameFromIndex(action)} with probability {probability}')