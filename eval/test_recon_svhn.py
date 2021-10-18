from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import os
import importlib
import numpy as np
import random
lr = 1e-4
latent_size = 256
num_epochs = 100
cuda_device = "1"
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def tocuda(x):
    if use_cuda:
        return x.cuda()
    return x

#torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--epochlist',nargs ="*", default = [],required=True)
parser.add_argument('--save_result_dir', required=True)
parser.add_argument("--dirlist", nargs ="*", default = [])
parser.add_argument("--objlist", nargs ="*", default = [])
parser.add_argument("--stochlist", nargs ="*", default = [])
parser.add_argument("--tanhlist", nargs ="*", default = [])
parser.add_argument("--sample_size", type=int, default = 0)
parser.add_argument("--batch",type=int, default = 50)
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
opt = parser.parse_args()

dataset = opt.dataset
dataroot = opt.dataroot
use_cuda = opt.use_cuda
dirlist = opt.dirlist
objlist = opt.objlist
stochlist = opt.stochlist
tanhlist = opt.tanhlist
sample_size = opt.sample_size
batch_size = batch = 100
epochlist = opt.epochlist
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

def fill_out(img, index, v):
    img = img.clone()
    for i in range(batch_size):
        img[i,:,index[i][0]:index[i][1],index[i][2]:index[i][3]] = v
    return img

def randomrect():
    index = []
    for i in range(batch_size):
        h = random.randint(0, 64)
        w = random.randint(0, 64)
        h_low = random.randint(0, 64-h)
        w_low = random.randint(0, 64-h)
        index.append([h_low,h+h_low,w_low,w+w_low])
    return index

def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

from torchvision import transforms

from model import Model


def _infer(path_to_checkpoint_file, image):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cuda()

    with torch.no_grad():
       # transform = transforms.Compose([
        #    transforms.Resize([64, 64]),
            #ransforms.CenterCrop([54, 54]),
         #   transforms.ToTensor(),
        #    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
       # ])

        #image = Image.open(path_to_input_image)
        #image = image.convert('RGB')
        image = 2*image-1
        images = image.cuda()

        vectors = model.eval()(images)

        #length_prediction = length_logits.max(1)[1]
        #digit1_prediction = digit1_logits.max(1)[1]
        #digit2_prediction = digit2_logits.max(1)[1]
        #digit3_prediction = digit3_logits.max(1)[1]
        #digit4_prediction = digit4_logits.max(1)[1]
        #digit5_prediction = digit5_logits.max(1)[1]

        #print('length:', length_prediction.item())
        #print('digits:', digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item(), digit5_prediction.item())
        return vectors


path_to_checkpoint_file = opt.checkpoint
if opt.dataset == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=opt.dataroot, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)

elif opt.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ])),
            batch_size=batch_size, shuffle=True)
elif opt.dataset == 'celeba':
    train_loader = torch.utils.data.DataLoader(
        dataset = datasets.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                           ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset = datasets.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                           ])),
        batch_size=batch_size, shuffle=True)
else:
        raise NotImplementedError

#torch.cuda.empty_cache()
#for d,(x,y) in enumerate(test_loader):
       # break
#d_recon_inpaint = [d_inpaint2[:batch]]
for i in range(len(dirlist)):
    obj = objlist[i]
    if(obj=="4_log_SN_bat_linear_tanh"):
        from model_disc_4_SN_bat_linear_tanh import *
   
    netE = tocuda(Encoder(latent_size, boolean_string(stochlist[i])))
    netG = tocuda(Generator(latent_size))
   # torch.cuda.empty_cache()
    with torch.no_grad():
     loss1 = []
     loss2 = []
     for (data, target) in test_loader:
       d_real = Variable(tocuda(data))
       d_real_tanh = 2*d_real -1 
       d_recon = [d_real]
       vec_recon = [_infer(path_to_checkpoint_file, F.interpolate(d_real,54))]

      # netE.apply(weights_init)
      # netG.apply(weights_init)

       netE.load_state_dict(torch.load(dirlist[i]+'/netE_epoch_%s.pth' % (epochlist[i])))
       netG.load_state_dict(torch.load(dirlist[i]+'/netG_epoch_%s.pth' % (epochlist[i])))

       netE.eval()
       netG.eval()
      # for obj in gc.get_objects():
        # try:
           # if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #  print(type(obj), obj.size())
        # except:
           # pass
       if(tanhlist[i]=="True"):
          z_real,_,_,_ = netE(d_real_tanh)
          z = z_real[:,:latent_size]
          d_recon_i = (1+netG(z.view(d_real.size()[0],latent_size,1,1)))/2
          d_recon.append(d_recon_i[:batch])
          vec_recon.append(_infer(path_to_checkpoint_file, F.interpolate(d_recon_i,54)))
       else:
          z_real, _, _, _ = netE(d_real)
         # z_inpaint,_,_,_ = netE(d_inpaint)
          z = z_real[:,:latent_size]
         # z_inpaint = z_inpaint[:,:latent_size]
          d_recon_i = netG(z.view(d_real.size()[0],latent_size,1,1))
        #  d_recon_inpaint_i = netG(z_inpaint.view(batch_size,latent_size,1,1))
          d_recon.append(d_recon_i[:batch])
          vec_recon.append(_infer(path_to_checkpoint_file, F.interpolate(d_recon_i,54)))
       #   d_recon_inpaint.append(d_recon_inpaint_i[:batch])
       criterion = nn.MSELoss()
       criterion2 = nn.MSELoss()
       loss1.append(criterion(d_recon[0],d_recon[1]).cpu().data.numpy())
       loss2.append(criterion2(vec_recon[0],vec_recon[1]).cpu().data.numpy())
     print(np.mean(loss1))
     print(np.std(loss1))
     print(np.mean(loss2))
     print(np.std(loss2))

dir_string = ""
for i in dirlist:
    dir_string +=i
    dir_string +='_'

if batch <= 50:
     save = torch.cat(d_recon[:batch],dim=0)
     vutils.save_image(save.cpu().data, '%s/%s_%s.png' % (opt.save_result_dir,dir_string),nrow=batch)
     save = torch.cat(d_recon_inpaint[:batch],dim=0)
     vutils.save_image(save.cpu().data, '%s/%s_%s_inpaint.png' % (opt.save_result_dir,dir_string),nrow=batch)
