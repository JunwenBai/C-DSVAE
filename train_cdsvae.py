import json
import random
import PIL
import functools
import utils
import progressbar
import numpy as np
import os
import argparse
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model import CDSVAE
from loss import contrastive_loss
from mutual_info import logsumexp, log_density

parser = argparse.ArgumentParser()
parser.add_argument('--lr',      default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--nEpoch',   default=300, type=int, help='number of epochs to train for')
parser.add_argument('--seed',    default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval',  default=10, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')

parser.add_argument('--dataset',   default='Sprite', type=str, help='dataset to train')
parser.add_argument('--frames',    default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels',  default=3, type=int, help='number of channels in images')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')

parser.add_argument('--f_rnn_layers', default=1,  type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size',     default=256,type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim',        default=256,  type=int,help='dim of f')
parser.add_argument('--z_dim',        default=32,type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim',        default=128,type=int, help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--loss_recon',    default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note',    default='', type=str, help='appx note')
parser.add_argument('--weight_f',      default=1,    type=float,help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z',      default=1,    type=float,help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_c_aug',      default=1,    type=float,help='weighting on content contrastive loss')
parser.add_argument('--weight_m_aug',      default=1,    type=float,help='weighting on motion contrastive loss')
parser.add_argument('--gpu',           default='0',  type=str,help='index of GPU to use')
parser.add_argument('--sche',          default='cosine', type=str, help='scheduler')


opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
mse_loss = nn.MSELoss().cuda()


def train(x, label_A, label_D, c_aug, m_aug, model, optimizer, contras_fn, opt, mode="train"):
    if mode == "train":
        model.zero_grad()

    if isinstance(x, list):
        batch_size = x[0].size(0) 
        seq_len = x[0].size(1) 
    else:
        batch_size = x.size(0)
        seq_len = x.size(1)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(x)
    f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, _ = model(c_aug)
    _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, _ = model(m_aug)

    if opt.loss_recon == 'L2': 
        l_recon = F.mse_loss(recon_x, x, reduction='sum')
    else:
        l_recon = torch.abs(recon_x - x).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1])) 
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1])) 
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar) 
    z_prior_var = torch.exp(z_prior_logvar) 
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size

    batch_size, n_frame, z_dim = z_post_mean.size()

    mi_fz = torch.zeros((1)).cuda()
    if True:
        _logq_f_tmp = log_density(f.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, opt.f_dim),
                                  f_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim),
                                  f_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim))

        _logq_z_tmp = log_density(z_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim),
                                  z_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim),
                                  z_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim))
        _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)

        logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()

    con_loss_c = contras_fn(f_mean, f_mean_c)
    con_loss_m = contras_fn(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))
    
    loss = l_recon + kld_f*opt.weight_f + kld_z*opt.weight_z + mi_fz
    if opt.weight_c_aug:
        loss += con_loss_c*opt.weight_c_aug
    if opt.weight_m_aug:
        loss += con_loss_m*opt.weight_m_aug

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
    return [i.data.cpu().numpy() for i in [l_recon, kld_f, kld_z, con_loss_c, con_loss_m]]

def fix_seed(seed):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

def main(opt):
    name = 'CDSVAE_Sprite_epoch-{}_bs-{}_rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
           '-weight:kl_f={}-kl_z={}-c_aug={}-m_aug={}-{}-sche_{}-{}'.format(
               opt.nEpoch, opt.batch_size, opt.rnn_size, opt.g_dim, opt.f_dim, opt.z_dim, opt.lr,
               opt.weight_f, opt.weight_z, opt.weight_c_aug, opt.weight_m_aug,
               opt.loss_recon, opt.sche, opt.note)
    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
    os.makedirs(opt.log_dir, exist_ok=True)

    summary_dir = os.path.join('./summary/', opt.dataset, name)
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_dir)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    fix_seed(opt.seed)

    # model, optimizer and scheduler
    cdsvae = CDSVAE(opt)
    cdsvae = cdsvae.cuda()
    opt.optimizer = optim.Adam

    optimizer = opt.optimizer(cdsvae.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.sche == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)
    elif opt.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch//2, gamma=0.5)
    elif opt.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    # dataset
    train_data, test_data = utils.load_dataset(opt)
    N, seq_len, dim1, dim2, n_c = train_data.data.shape
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    opt.dataset_size = len(train_data)

    epoch_loss = Loss()
    contras_fn = contrastive_loss(tau=0.5, normalize=True)
    
    # training and testing
    cur_step = 0
    for epoch in range(opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        cdsvae.train()
        epoch_loss.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(max_value=len(train_loader)).start()
        for i, data in enumerate(train_loader):
            progress.update(i+1)
            x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
            x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()

            recon, kld_f, kld_z, con_loss_c, con_loss_m = train(x, label_A, label_D, c_aug, m_aug, cdsvae, optimizer, contras_fn, opt)
            epoch_loss.update(recon, kld_f, kld_z, con_loss_c, con_loss_m)
            
            lr = optimizer.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar("lr", lr, cur_step)
                writer.add_scalar("Train/mse", recon.item(), cur_step)
                writer.add_scalar("Train/kld_f", kld_f.item(), cur_step)
                writer.add_scalar("Train/kld_z", kld_z.item(), cur_step)
                writer.add_scalar("Train/con_loss_c", con_loss_c.item(), cur_step)
                writer.add_scalar("Train/con_loss_m", con_loss_m.item(), cur_step)
                cur_step += 1

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        lr = optimizer.param_groups[0]['lr']
        print('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_loss_c: %.5f |'
                  ' con_loss_m: %.5f | lr: %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4], lr))

        if epoch == opt.nEpoch-1 or epoch % opt.evl_interval == 0:
            cdsvae.eval()

            test_mse = test_kld_f = test_kld_z = test_c_loss = test_m_loss = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
                x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()

                with torch.no_grad():
                    recon, kld_f, kld_z, con_loss_c, con_loss_m = train(x, label_A, label_D, c_aug, m_aug, cdsvae, optimizer, contras_fn, opt, mode="test")

                test_mse += recon
                test_kld_f += kld_f
                test_kld_z += kld_z
                test_c_loss += con_loss_c
                test_m_loss += con_loss_m

            n_batch = len(test_loader)
            print('[%02d] Val recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_loss_c: %.5f | con_loss_m: %.5f' % 
                (epoch, test_mse.item()/n_batch, test_kld_f.item()/n_batch, test_kld_z.item()/n_batch, test_c_loss.item()/n_batch, test_m_loss.item()/n_batch))

            n_batch = len(test_loader)
            if writer is not None:
                writer.add_scalar("Val/mse", test_mse.item()/n_batch, epoch)
                writer.add_scalar("Val/kld_f", test_kld_f.item()/n_batch, epoch)
                writer.add_scalar("Val/kld_z", test_kld_z.item()/n_batch, epoch)
                writer.add_scalar("Val/con_loss_c", test_c_loss.item()/n_batch, epoch)
                writer.add_scalar("Val/con_loss_m", test_m_loss.item()/n_batch, epoch)

            torch.save({
                'model': cdsvae.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (opt.log_dir, epoch))

def reorder(sequence):
    return sequence.permute(0,1,4,2,3)

class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, recon, kld_f, kld_z, con_loss_c, con_loss_m):
        self.recon.append(recon)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)
        self.con_loss_c.append(con_loss_c)
        self.con_loss_m.append(con_loss_m)

    def reset(self):
        self.recon = []
        self.kld_f = []
        self.kld_z = []
        self.con_loss_c = []
        self.con_loss_m = []

    def avg(self):
        return [np.asarray(i).mean() for i in
                [self.recon, self.kld_f, self.kld_z, self.con_loss_c, self.con_loss_m]]

if __name__ == '__main__':
    main(opt)


