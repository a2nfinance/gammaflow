import os
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import imageio
import numpy as np

np.random.seed(777)
torch.manual_seed(777)

cuda = False
img_size = 96
nc = 3 # number of chanel
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M

class GRU(nn.Module):
    def __init__(self, input_size = 10, hidden_size = 10, gpu=True):
        super(GRU, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.GRUCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant_(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant_(b_hz, init_forget_bias)
            else:
                init.constant_(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()

current_path = os.path.dirname(__file__)            
trained_path = os.path.join(current_path, 'trained_models')
gru = GRU(d_E, hidden_size, gpu=cuda)

if cuda:
    gru.load_state_dict(torch.load(trained_path + '/GRU_epoch-120000.model'), strict=False)
else:
    gru.load_state_dict(torch.load(trained_path + '/GRU_epoch-120000.model', map_location=torch.device('cpu')), strict=False)

#gru.initWeight()

def gen_z(n_frames, batch_size = 16):
    #print("----Generating Z-----")
    #print(f"N_FRAMES: {n_frames}")
    #print(f"BATCH_SIZE: {batch_size}")
    #print(f"D_C: {d_C}")
    #print(f"D_E: {d_E}")
    #print(f"nz: {nz}")
    np.random.seed(777)
    torch.manual_seed(777)

    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, n_frames).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    #print("----End Generating Z-----")
    return z.view(batch_size, n_frames, nz, 1, 1)

def getNumFrames(reader):
    
    try:
        return math.ceil(reader.get_meta_data()['fps'] * reader.get_meta_data()['duration'])
    
    except AttributeError as _:
        filename = reader
        return getNumFrames(imageio.get_reader(filename,  'ffmpeg'))

def readVideoImageio(filename, n_channels= 3):
    
    frames = []
    with imageio.get_reader(filename,  'ffmpeg') as reader:
    
        shape = reader.get_meta_data()['size']
        
        for img in reader:
            try:
                frames.append(img)
                
            except Error as err:
                print(err)
         
    # Paranoid double check
    if not reader.closed:
        reader.close()
        
    videodata = np.array(frames, dtype= np.uint8)
            
    return videodata

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    extensions = list(extensions)
    
    returnValue = False
    for extension in extensions:
        returnValue = returnValue or filename.lower().endswith(extension)
    
    return returnValue


def make_dataset(dir, class_to_idx, extensions=None, classes = []):
    videos = []

    if os.path.isdir(dir):

        dir = os.path.expanduser(dir)
        
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
            
        for target in sorted(class_to_idx.keys()):
            
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if len(classes) > 0:
                    if not any([element.lower() in root.lower() for element in classes]):
                        continue

                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        videos.append(item)

    else: # It is a file containing preprocessed informations.
        with open(dir, 'r') as file:
            for line in file:
                if line == '':
                    continue
                line      = line.rstrip('\n\r')
                target    = os.path.split( os.path.split(line)[0] ) [1]
                path      = os.path.join( os.path.split(dir)[0], line)
                item      = (path, class_to_idx[target])
                videos.append(item)
                    
    return videos

def trim(video):
    start = 0 
    end = video.shape[1] - 1
    return video[:, start:end, :, :]

def convert_class(cl):
    switcher={
                1:2,
                2:5,
                3:4,
                4:2,
                5:1,
                6:10,
                7:3,
                8:6,
                9:8,
                10:7
             }
    return switcher.get(cl, "Invalid class")