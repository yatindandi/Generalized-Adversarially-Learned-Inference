
import torch
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import os
import importlib
import numpy as np
import random
from torch.autograd import Function
import torchvision.utils as vutils
from model_disc_4_celeb_SN_bat_linear_tanh_extra import *
#import os

batch_size = 20
lr = 4e-5
latent_size = 256
num_epochs = 50
#cuda_device = "1"

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)
parser.add_argument('--last_epoch',type=int, default=0)
#parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)

def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

from torchvision import transforms
from inception import InceptionV3
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
print(block_idx)
model = InceptionV3([block_idx])

path_to_checkpoint_file = 'fsa'

tocuda(model)
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

model.apply(set_bn_eval)
#model.apply(set_bn_to_eval)r
def _infer(path_to_checkpoint_file, image):
        image = image
        images = image.cuda()
        vectors = model(images)[0]
          
        return vectors



if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='extra', download=True,
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
elif opt.dataset == 'celeba':
    train_loader = torch.utils.data.DataLoader(
        dataset = datasets.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                           ])),
        batch_size=batch_size, shuffle=True)
else:
    raise NotImplementedError

netE = nn.DataParallel(tocuda(Encoder(latent_size, True)))
netG = nn.DataParallel(tocuda(Generator(latent_size)))
netD = nn.DataParallel(tocuda(Discriminator(latent_size, 0.2, 4)))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

    
optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5,0.999))

if(opt.last_epoch!=0):
    netE.load_state_dict(torch.load(opt.save_model_dir+'/netE_epoch_%d.pth' %(opt.last_epoch)))
    netG.load_state_dict(torch.load(opt.save_model_dir+'/netG_epoch_%d.pth' %(opt.last_epoch)))
    netD.load_state_dict(torch.load(opt.save_model_dir+'/netD_epoch_%d.pth' %(opt.last_epoch)))
    optimizerG.load_state_dict(torch.load(opt.save_model_dir+'/optG_epoch_%d.pth' %(opt.last_epoch)))
    optimizerD.load_state_dict(torch.load(opt.save_model_dir+'/optD_epoch_%d.pth' %(opt.last_epoch)))

def train(iter,epoch, loss):
        if (iter+1)%6==0:
            optimizer = optimizerG
        else:
            optimizer = optimizerD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iter == 0):
            scheduler_g.step()
            scheduler_d.step()
start_epoch = opt.last_epoch+1

criterion = nn.CrossEntropyLoss()
criterion_D = nn.CrossEntropyLoss()

#criterion_warmup = nn.BCELoss()

for epoch in range(start_epoch,num_epochs+1):

    i = 0
    for (data, target) in train_loader:

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        real_label_1 = Variable(tocuda(torch.zeros(batch_size).type(torch.LongTensor)))
        real_label_2 = Variable(tocuda(torch.zeros(batch_size).type(torch.LongTensor))) + 1
        fake_label_1 = Variable(tocuda(torch.zeros(batch_size).type(torch.LongTensor))) + 2
        fake_label_2 = Variable(tocuda(torch.zeros(batch_size).type(torch.LongTensor))) + 3

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 1 and i == 0:
            netG.output_bias.data = 0.5*get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = 2*Variable(tocuda(data))-1

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + sigma*epsilon

        y_real = _infer(path_to_checkpoint_file, (1+d_real)/2)
        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1), y_real.view(batch_size, 2048, 1, 1))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        recon_real = netG(output_z.view(batch_size, latent_size, 1, 1))
        enc_recon_real, _, _, _ = netE(recon_real)

        enc_recon_real = enc_recon_real.view(batch_size, -1)

        mu_recon_real, log_sigma_recon_real = enc_recon_real[:, :latent_size], enc_recon_real[:, latent_size:]
        sigma_recon_real = torch.exp(log_sigma_recon_real)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z_recon_real = mu_recon_real + sigma_recon_real*epsilon
        y_recon = _infer(path_to_checkpoint_file, (1+recon_real)/2)
        output_real_2, _ = netD(d_real + noise1, output_z_recon_real.view(batch_size, latent_size, 1, 1),y_recon.view(batch_size, 2048, 1, 1))
        y_fake = _infer(path_to_checkpoint_file, (1+d_fake)/2)
        output_fake, _ = netD(d_fake + noise2, z_fake, y_fake.view(batch_size, 2048, 1, 1))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        enc_fake, _, _, _  = netE(d_fake)
        enc_fake = enc_fake.view(batch_size, -1)
        mu_fake, log_sigma_fake = enc_fake[:, :latent_size], enc_fake[:, latent_size:]
        sigma_fake = torch.exp(log_sigma_fake)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z_fake = mu_fake + sigma_fake*epsilon

        recon_fake = netG(output_z_fake.view(batch_size, latent_size, 1, 1))
        output_fake_2, _ = netD(recon_fake + noise2, z_fake, y_fake.view(batch_size, 2048, 1, 1))
 

        o_real = output_real
        o_real2 = output_real_2
        o_fake = output_fake
        o_fake2 = output_fake_2

        loss_d = 0.5*(criterion_D(o_real, real_label_1) + criterion_D(o_real2, real_label_2) + criterion_D(o_fake, fake_label_1) + criterion_D(o_fake2, fake_label_2))
        
        loss_g = (1/6)*(criterion(o_real, real_label_2) + criterion(o_real, fake_label_1) + criterion(o_real, fake_label_2) + criterion(o_real2, real_label_1) + criterion(o_real2, fake_label_1) + criterion(o_real2, fake_label_2) + criterion(o_fake, real_label_1) + criterion(o_fake, real_label_2)+ criterion(o_fake, fake_label_2)+ criterion(o_fake2, real_label_1) + criterion(o_fake2, real_label_2)+ criterion(o_fake2, fake_label_1))


        optimizerD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        o_real = F.softmax(output_real)
        o_real2 = F.softmax(output_real_2)
        o_fake = F.softmax(output_fake)
        o_fake2 = F.softmax(output_fake_2)
        
        d_real = (d_real+1)/2
        d_fake = (d_fake+1)/2

        if i % 1 == 0:
            #print("Epoch :", epoch, "Iter :", i, "D Loss :{:.4f}".format(loss_d.data.item()), "G loss :{:.4f}".format(loss_g.data.item()),
            #      "D(x,E(x)) :{:.4f}".format(o_real[:,0].mean().data.item()),"D(x,E(G(E(x)))) :{:.4f}".format(o_real2[:,0].mean().data.item()), "D(G(z),z) :{:.4f}".format(o_fake[:,0].mean().data.item()), "D(G(E(G(z))),z) :{:.4f}".format(o_fake2[:,0].mean().data.item()))
            print("Epoch :", epoch, "Iter :", i, "D Loss :{:.4f}".format(loss_d.data.item()), "G loss :{:.4f}".format(loss_g.data.item()),
                  "D(x,E(x)) :{:.2f},{:.2f},{:.2f},{:.2f},".format(o_real[:,0].mean().data.item(),o_real[:,1].mean().data.item(),o_real[:,2].mean().data.item(),o_real[:,3].mean().data.item()),"D(x,EGE(x)) :{:.2f},{:.2f},{:.2f},{:.2f},".format(o_real2[:,0].mean().data.item(),o_real2[:,1].mean().data.item(),o_real2[:,2].mean().data.item(),o_real2[:,3].mean().data.item()), "D(G(z),z) :{:.2f},{:.2f},{:.2f},{:.2f},".format(o_fake[:,0].mean().data.item(),o_fake[:,1].mean().data.item(),o_fake[:,2].mean().data.item(),o_fake[:,3].mean().data.item()), "D(GEG(z),z) :{:.2f},{:.2f},{:.2f},{:.2f},".format(o_fake2[:,0].mean().data.item(),o_fake2[:,1].mean().data.item(),o_fake2[:,2].mean().data.item(),o_fake2[:,3].mean().data.item()))

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], '%s/fake.png' % (opt.save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], '%s/real.png'% (opt.save_image_dir))

        i += 1

    if epoch % 10 == 0 :

        vutils.save_image(d_fake.cpu().data[:16, ], '%s/fake_%d.png' % (opt.save_image_dir, epoch))


    if (epoch % 10 == 0) :
        file = open(opt.save_model_dir+'/epoch_'+str(0)+'.txt','w')
        sv = "Epoch :" +str(epoch)+ ",Iter :"+ str(i-1) +  ",D Loss :{:.4f}".format(loss_d.data.item())+ ",G loss :{:.4f}".format(loss_g.data.item())
        file.write(sv) 
        file.close() 
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.save_model_dir,epoch))
        torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.save_model_dir,epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.save_model_dir,epoch))
        torch.save(optimizerG.state_dict(),'%s/optG_epoch_%d.pth' % (opt.save_model_dir,epoch))
        torch.save(optimizerD.state_dict(),'%s/optD_epoch_%d.pth' % (opt.save_model_dir,epoch))
