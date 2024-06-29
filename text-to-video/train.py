# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Lambda, Compose

from models import VideoDiscriminator, VideoGenerator, GRU


parser = argparse.ArgumentParser(description='Start trainning GammaFlow.....')
parser.add_argument('--cuda', type=bool, default=False,
                     help='Set to use the GPU.')
parser.add_argument('--ngpu', type=int, default=1,
                     help='set the number of gpu you use')
parser.add_argument('--batch_size', type=int, default=16,
                     help='set batch_size, default: 16')
parser.add_argument('--n_epochs', type=int, default=120000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre_train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
parser.add_argument('--lr', type=float, default=0.0002,
                     help='set learning rate, default: 0.0002')
parser.add_argument('--trim_video', type=int, default=16,
                     help='set number of frames, default: 16')

EXPLORATORY_DATA_ANALYSIS = False

parser.add_argument('--i_epochs_checkpoint', type=int, default=1,
                     help='set num of epochs between checkpoints, default: 1')
parser.add_argument('--i_epochs_saveV', type=int, default=1,
                     help='set num of epochs between save fake video, default: 1')
parser.add_argument('--i_epochs_display', type=int, default=1,
                     help='set num of epochs between print information, default: 1')

args       = parser.parse_args()
cuda       = args.cuda
ngpu       = args.ngpu
batch_size = args.batch_size
n_epochs     = args.n_epochs
pre_train  = args.pre_train
trim_video  = args.trim_video
lr = args.lr

## Addition for training 
n_epochs_saveV      = args.i_epochs_saveV
n_epochs_display    = args.i_epochs_display
n_epochs_check      = args.i_epochs_checkpoint
max_frame           = 25
#### End of additions

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

if cuda:
    torch.cuda.set_device(0)


''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*/*')

#transformation = Compose([ToTensor(), Lambda(lambda tensor: (tensor - tensor.mean() )/ tensor.std())])

transformation = Compose([Lambda(lambda video: video.transpose(3, 0, 1, 2)/255.0),
                            Lambda(lambda video: video[ : , : max_frame, :, : ]),
                            Lambda(lambda video: torch.FloatTensor(video))])

filenameDictClassesIdx = "classInd.txt"
dictClassesIdx = {}

with open(os.path.join(current_path, "classes", filenameDictClassesIdx)) as file:
    for line in file:
        dictClassesIdx[ line.split() [1]] = int( line.split() [0] )

dataset = DatasetFolder(resized_path, skvideo.io.vread, ["mp4"], transform= transformation)
dataset.class_to_idx = dictClassesIdx
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle= True, num_workers= 0, pin_memory= True, drop_last= True)

if (EXPLORATORY_DATA_ANALYSIS):

    minimum = 12000
    #for line in files:
        #print (f"Started loading one of {len(files)} videos into memory... It will be fast.")
    video = skvideo.io.vread(files[0])
    print(str(video.shape))
    video = video.transpose(3, 0, 1 ,2) / 255.0
    video = torch.FloatTensor(video)
    print(str(video.shape))
    #print (f"Ended transforming into tensor: size is { (video.nelement() * video.element_size()) / (1024 * 1024)} MB")
    #print (f"Final memory would be about {(video.nelement() * video.element_size()) / (1024 * 1024 * 1024) * len(files)} GB")
    # transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
    if (video.shape[0] < minimum ):
        minimum = video.shape[0] #Minimum is 180


    print(minimum)

''' prepare video sampling '''

n_videos = len(dataset) #len(videos)
T = trim_video

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]

# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    #print("-----TRIMMING NOISE-----")
    #print(f"Noise Size: {noise.size()}")
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    #print("-----END OF TRIMMING NOISE-----")
    return noise[:, start:end, :, :, :]


# video length distribution
#video_lengths = [video.shape[1] for video in videos]


''' set models '''

img_size = 96
nc = 3 # number of chanel
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M
criterion = nn.BCELoss()

dis = VideoDiscriminator(nc, ndf, T=T, ngpu=ngpu)
gen = VideoGenerator(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()

''' prepare for train '''

#label = torch.FloatTensor()

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models')
def checkpoint(model, optimizer, epoch):
    filename = os.path.join(trained_path, '%s_epoch-%d' % (model.__class__.__name__, epoch))
    torch.save(model.state_dict(), filename + '.model')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)


''' adjust to cuda '''

if cuda:
    dis.cuda()
    gen.cuda()
    gru.cuda()
    criterion.cuda()

# setup optimizer
lr = 0.0002
betas=(0.5, 0.999)
optim_D  = optim.Adam(dis.parameters(), lr=lr, betas=betas)
optim_G  = optim.Adam(gen.parameters(), lr=lr, betas=betas)
optim_GRU = optim.Adam(gru.parameters(),   lr=lr, betas=betas)

''' use pre-trained models '''

if pre_train == True:
    if cuda:
        dis.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoDiscriminator_epoch-120000.model'), strict=False)
        gen.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoGenerator_epoch-120000.model'), strict=False)
        gru.load_state_dict(torch.load(trained_path + '/pre_trained_models/GRU_epoch-120000.model'), strict=False)
        optim_D.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoDiscriminator_epoch-120000.state'), strict=False)
        optim_G.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoGenerator_epoch-120000.state'), strict=False)
        optim_GRU.load_state_dict(torch.load(trained_path + '/pre_trained_models/GRU_epoch-120000.state'), strict=False)
    else:
        dis.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoDiscriminator_epoch-120000.model', map_location=torch.device('cpu')), strict=False)
        gen.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoGenerator_epoch-120000.model', map_location=torch.device('cpu')), strict=False)
        gru.load_state_dict(torch.load(trained_path + '/pre_trained_models/GRU_epoch-120000.model', map_location=torch.device('cpu')), strict=False)
        optim_D.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoDiscriminator_epoch-120000.state', map_location=torch.device('cpu')), strict=False)
        optim_G.load_state_dict(torch.load(trained_path + '/pre_trained_models/VideoGenerator_epoch-120000.state', map_location=torch.device('cpu')), strict=False)
        optim_GRU.load_state_dict(torch.load(trained_path + '/pre_trained_models/GRU_epoch-120000.state', map_location=torch.device('cpu')), strict=False)


''' calc grad of models '''

def bp(inputs, y, retain=False):
    #print("----BackPropagate_V-----")
    #print(inputs.size())
    if cuda:
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


''' gen input noise for fake video '''

def gen_z(n_frames, batch_size = batch_size):
    #print("----Generating Z-----")
    #print(f"N_FRAMES: {n_frames}")
    #print(f"BATCH_SIZE: {batch_size}")
    #print(f"D_C: {d_C}")
    #print(f"D_E: {d_E}")
    #print(f"nz: {nz}")
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


''' train models '''

start_time = time.time()

print(f"Starting training: CUDA is { 'On' if cuda else 'Off'}")

for epoch in range(1, n_epochs+1):
    ''' prepare real images '''
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)

    # Get data iterator
    data_iter = iter(dataloader) #Iterator
    data_len = len(dataloader) #Num Batches
    data_i = 0

    processedClass = None

    while data_i < data_len:

        try:
            (real_videos, labels) = next(data_iter) #random_choice()

            for (key, val) in dictClassesIdx.items():
                if ( val in labels.tolist() ):
                    pass

            if cuda:
                real_videos = real_videos.cuda()

            real_videos = Variable(real_videos)

            ''' prepare fake videos'''
            # note that n_frames is sampled from video length distribution
            n_frames = T + 2 + np.random.randint(0, real_videos.size()[2]) #video_lengths[np.random.randint(0, n_videos)]
            # print("n_videos", real_videos.size()[2])
            Z = gen_z(n_frames, batch_size)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
            # trim => (batch_size, T, nz, 1, 1)
            Z = trim_noise(Z)
            # generate videos
            Z = Z.contiguous().view(batch_size*T, nz, 1, 1)
            fake_videos = gen(Z, labels)
            fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
            # transpose => (batch_size, nc, T, img_size, img_size)
            fake_videos = fake_videos.transpose(2, 1)

            ''' train discriminators '''
            # video
            dis.zero_grad()
            randomStartFrameIdx = np.random.randint(0, real_videos.size()[2] - T - 1)
            #print("-----INFOS-----")
            #print(f"RandomStartFrame:{randomStartFrameIdx}")
            #print(f"Video Size:{real_videos.size()}")
            #print("-----END OF INFOS-----")
            croppedRealVideos = real_videos[:,:,randomStartFrameIdx: randomStartFrameIdx + T, :, :]
            #err_Dv_real, Dv_real_mean = bp_v(croppedRealVideos, 0.9)

            err_D_real, D_real_mean = bp(croppedRealVideos, labels.type(torch.FloatTensor) / len(dictClassesIdx))
            err_D_fake, D_fake_mean = bp(fake_videos.detach(), 0)
            err_D = err_D_real + err_D_fake
            optim_D.step()

            ''' train generators '''
            gen.zero_grad()
            gru.zero_grad()
            # video. notice retain=True for back prop twice
            err_G, _ = bp(fake_videos, labels.type(torch.FloatTensor) / len(dictClassesIdx), retain=True)
            optim_G.step()
            optim_GRU.step()

            '''Increment index for Batch'''
            data_i = data_i + 1
        except StopIteration:
            break

        except KeyboardInterrupt:
            save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)
            checkpoint(dis, optim_D, epoch)
            checkpoint(gen, optim_G, epoch)
            checkpoint(gru,   optim_GRU, epoch)


    if epoch % n_epochs_display == 0:
        print('[%d/%d] (%s) Loss_D: %.4f Loss_G: %.4f  D_real_mean %.4f D_fake_mean %.4f'
              % (epoch, n_epochs, timeSince(start_time), err_D, err_G, D_real_mean, D_fake_mean))

    if epoch % n_epochs_saveV == 0:
        save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)

    if epoch % n_epochs_check == 0:
        checkpoint(dis, optim_D, epoch)
        checkpoint(gen, optim_G, epoch)
        checkpoint(gru, optim_GRU, epoch)
