# coding: utf-8

import os
import math
import imageio
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

class Debug(nn.Module):
    def forward(self, input):
        print(input.shape, flush = True)
        return input


class Noise(nn.Module):
    def __init__(self, use_noise, sigma = 0.2, use_gpu = False):
        super(Noise, self).__init__()
        self.sigma      = sigma
        self.use_gpu    = use_gpu
        self.use_noise  = use_noise
        
        
    def forward(self, arg):
        
        if self.use_gpu and self.use_noise:
            return arg + self.sigma * torch.cuda.FloatTensor(arg.size(), requires_grad = False).normal_()
        
        elif self.use_noise:
            return arg + self.sigma * torch.FloatTensor(arg.size(), requires_grad = False).normal_()
        
        else:
            return arg


class VideoDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1, nClasses= 11):
        super(VideoDiscriminator, self).__init__()
        self.ngpu       = ngpu
        self.nClasses   = nClasses
        
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            Flatten(),
            nn.Linear(int((ndf*8)*(T/16)*6*6), 1 + nClasses),
            nn.Sigmoid()
        )

    def forward(self, input):
    
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            
        else:
            output = self.main(input)
            
        labels = output[:, 1 : output.size(1)]
        output = output[:, 0]
        result = output.view(-1, 1).squeeze(1)

        return result, labels


# see: _netG in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class VideoGenerator(nn.Module):
    '''
        Constructor
        -----------
        The constructor of Generator_I takes 6 arguments, all optional.
        
        nc:         integer, default= 3
            Num channels of the image to produce.
        
        ngf:        integer, default= 64
            Parameter of the ConvTranspose2d Layers.
        
        nz:         integer, default= 60
            Number of samples for the noise.
        
        cuda:       bolean, default= False
            Using cuda or not.
            
        ngpu:       integer, default= 1
            Number of GPU on which the model will run.
            
        nClasses:   integer, default= 102
            Number of classes on which the Embedding module will work.
            
        batch_size: integer, default = 16
            Batch size for each argument that will be passed to the model.
            
    
    '''
    
    def __init__(self, nc=3, ngf=64, nz=60, cuda = False, ngpu=1, nClasses= 11, batch_size= 16):
        super(VideoGenerator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.gpu = cuda
    
        # Addition for Conditioning the Model
        # nClasses = #Action Class + 1 (Fake Class) 
        self.label_sequence = nn.Sequential(
            nn.Embedding(nClasses, nClasses//batch_size),
            nn.Linear(nClasses//batch_size, nz),
            nn.ReLU(True)
        )
        
        self.combine_sequence = nn.Sequential(
            nn.Linear(ngf*4 + batch_size, ngf*4)
        )
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 24 x 24
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input, labels):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # Addition to prepare labels to be concatenated with input.
            labels = nn.parallel.data_parallel(self.label_sequence, labels, range(self.ngpu))
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            
            input = nn.parallel.data_parallel(self.combine_sequence, combinedInput, range(self.ngpu)).transpose(3,0)
            
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            
        else:
            labels = self.label_sequence(labels)
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            
            input = self.combine_sequence(combinedInput).transpose(3,0)
            
            output = self.main(input)
            
        return output

    def sample_z_categ(self, num_samples, video_len, category = None):
        video_len = video_len if video_len is not None else 16

        if category:
            classes_to_generate = np.array(category)

        else:
            classes_to_generate = np.random.randint(self.nz, size=num_samples)

        one_hot = np.zeros((num_samples, self.nz), dtype=np.float32)
        print(one_hot)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        print(one_hot)
        one_hot_video = np.repeat(one_hot, video_len, axis=0)
        print(one_hot_video)
        one_hot_video = torch.from_numpy(one_hot_video)

        if self.gpu:
            one_hot_video = one_hot_video.cuda()
        return Variable(one_hot_video), classes_to_generate

    def sample_videos(self, num_samples, video_len=None, category = None):
        n_channels = 3
        video_len = video_len if video_len is not None else 16
         # Sample a single latent vector for each video
         # Sample a single latent vector for each video and create variations
        base_z = torch.randn(num_samples, self.nz)
        if self.gpu:
            base_z = base_z.cuda()
        
        # Create variations around the base_z to generate movement
        z = []
        for i in range(num_samples):
            variations = self.create_variations(base_z[i], video_len)
            z.append(variations)

        # Convert list to tensor
        z = torch.stack([torch.stack(z_i) for z_i in z]).view(-1, self.nz)

        _, z_category_labels = self.sample_z_categ(num_samples, video_len, category)
        h = self.main(z.view(z.size(0), z.size(1), 1, 1))

        h = h.view( int( h.size(0) / video_len), video_len, n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if self.gpu:
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, z_category_labels

    def create_variations(self, base_z, num_steps, variation_scale=0.2):
        # Create slight variations around the base_z
        return [base_z + variation_scale * torch.randn_like(base_z) for _ in range(num_steps)]


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


''' utils '''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
   

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
