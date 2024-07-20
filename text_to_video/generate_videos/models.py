##############################################################################################
## The model source code is referenced from MoCoGAN                                         ##
## (see https://github.com/CarloP95/mocogan/tree/a71449c0b617265b8c5193449b8121267941bf4c), ##
## where we added the sigmoid activation function to the VideoGenerator function.           ##
## Additionally, we improved the sample_videos to call the GRU pre-trained model.           ##
##############################################################################################

# coding: utf-8

import sys
import os
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from utils import gen_z, convert_class

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
            nn.Linear(int((ndf*8)*(T/16)*6*6), nClasses),
            nn.Sigmoid()
        )

    def forward(self, input):
    
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, device_ids=range(self.ngpu))
            
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
        
        cuda:       boolean, default= False
            Using cuda or not.
            
        ngpu:       integer, default= 1
            Number of GPU on which the model will run.
            
        nClasses:   integer, default= 11
            Number of classes on which the Embedding module will work.
            
        batch_size: integer, default = 16
            Batch size for each argument that will be passed to the model.
            
    
    '''
    
    def __init__(self, nc=3, ngf=64, nz=60, cuda = False, ngpu=1, nClasses= 11, batch_size= 16):
        super(VideoGenerator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.gpu = cuda
        self.batch_size = batch_size
        self.nClasses = nClasses
        # Addition for Conditioning the Model
        # nClasses = Action Class + 1 (fake class)
        self.label_sequence = nn.Sequential(
            nn.Embedding(nClasses, nClasses//16),
            nn.Linear(nClasses//16, nz),
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
        if isinstance(input, torch.Tensor) and self.ngpu > 1:
            # Addition to prepare labels to be concatenated with input.
            labels = nn.parallel.data_parallel(self.label_sequence, labels, device_ids=range(self.ngpu))
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            
            input = nn.parallel.data_parallel(self.combine_sequence, combinedInput, device_ids=range(self.ngpu)).transpose(3,0)
            
            output = nn.parallel.data_parallel(self.main, input, device_ids=range(self.ngpu))
            
        else:
            labels = self.label_sequence(labels)
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            input = self.combine_sequence(combinedInput).transpose(3,0)
            output = self.main(input)
        return output
    
    def sample_videos(self, video_len=None, category = None):

        # n_channels = 3
        if category:
            z_category_labels = np.array([category for i in range(self.batch_size)])
        else:
            z_category_labels = np.random.randint(self.nClasses, size=self.batch_size)

        z_category_labels = torch.from_numpy(z_category_labels).long()

        # Use nn.Embedding to embed the label radiation component vector
        if self.gpu:
            z_category_labels = z_category_labels.cuda()

        labels = self.label_sequence(z_category_labels)  # Map labels to embedding vectors
        
        labels = labels.view(self.batch_size, self.nz, 1, 1)  # Reshape into (batch_size, nz, 1, 1)


        if self.gpu:
            labels = labels.cuda()

        video_len = video_len if video_len is not None else 16


        # Create noise in the pre_trained model
        z = gen_z(video_len, 16)

        input = z.contiguous().view(16, video_len, self.nz, 1, 1)
        id = convert_class(category)
        input = input[id-1:id, :, :, :, :]
        # Reshape to size: (bach_size*video_len, nz, 1, 1)
        input = input.view(self.batch_size*video_len, self.nz, 1, 1)

        combinedInput = torch.cat((input, labels), 0)

        h = self.main(combinedInput)
        #h = h.view( int( h.size(0) / video_len), video_len, n_channels, h.size(3), h.size(3))
        h = h.unsqueeze(0)
        #h = trim(h)
        h = h.permute(0, 2, 1, 3, 4)

        return h
    
    def generate_noise(self, video_len=None, category = None):
        if category:
            z_category_labels = np.array([category for i in range(self.batch_size)])
        else:
            z_category_labels = np.random.randint(self.nClasses, size=self.batch_size)

        z_category_labels = torch.from_numpy(z_category_labels).long()

        # Use nn.Embedding to embed the label radiation component vector
        if self.gpu:
            z_category_labels = z_category_labels.cuda()


        video_len = video_len if video_len is not None else 16


        # Create noise in the pre_trained model
        z = gen_z(video_len, 16)

        input = z.contiguous().view(16, video_len, self.nz, 1, 1)
        id = convert_class(category)
        input = input[id-1:id, :, :, :, :]
        # Reshape to size: (bach_size*video_len, nz, 1, 1)
        input = input.view(self.batch_size*video_len, self.nz, 1, 1)
        # print("input", input.size())

        return input, z_category_labels


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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
   
