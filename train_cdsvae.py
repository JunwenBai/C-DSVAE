import json
import random
import functools
import PIL 
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

from mutual_info import logsumexp, log_density, log_importance_weight_matrix

from model import CDSVAE, classifier_Sprite_all
from loss import contrastive_loss, compute_mi

from torch.utils.tensorboard import SummaryWriter
from utils import entropy_Hy, entropy_Hyx, inception_score, KL_divergence


parser = argparse.ArgumentParser()
parser.add_argument('--lr',      default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--log_dir', default='./logs_sprite', type=str, help='base directory to save logs')
parser.add_argument('--model_dir', default='', type=str, help='model to load or resume')
parser.add_argument('--data_root', default='./data', type=str, help='root directory for data')
parser.add_argument('--nEpoch',   default=300, type=int, help='number of epochs to train for')
parser.add_argument('--seed',    default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval',  default=10, type=int, help='evaluate every n epoch')

parser.add_argument('--dataset',   default='Sprite', type=str, help='dataset to train')
parser.add_argument('--frames',    default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels',  default=3, type=int, help='number of channels in images')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1,  type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size',     default=256,type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim',        default=256,  type=int,help='dim of f')
parser.add_argument('--z_dim',        default=32,type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim',        default=128,type=int, help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--type_gt',  type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--loss_recon',    default='L2', type=str, help='reconstruction loss: L1, L2')
parser.add_argument('--note',    default='S3', type=str, help='appx note')
parser.add_argument('--weight_f',      default=1,    type=float,help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z',      default=1,    type=float,help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_c_aug',      default=1,    type=float,help='weighting on content contrastive loss')
parser.add_argument('--weight_m_aug',      default=1,    type=float,help='weighting on motion contrastive loss')
parser.add_argument('--gpu',           default='0',  type=str,help='index of GPU to use')
parser.add_argument('--sche',          default='cosine', type=str, help='scheduler')


opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

mse_loss = nn.MSELoss().cuda()
#triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
#CE_loss = nn.CrossEntropyLoss().cuda()

# --------- training funtions ------------------------------------
def train(x, label_A, label_D, c_aug, m_aug, model, optimizer, contras_fn, opt, mode="train"):
    if mode == "train":
        model.zero_grad()
    #print("x:", len(x)) # 3
    #print("x[0]:", x[0].shape) # [128, 8, 3, 64, 64]

    if isinstance(x, list):
        batch_size = x[0].size(0) # 128
        seq_len = x[0].size(1) # 8
    else:
        batch_size = x.size(0)
        seq_len = x.size(1)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(x) #pred
    f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, _ = model(c_aug)
    _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, _ = model(m_aug)

    #print("f_mean:", len(f_mean), f_mean[0].shape) # 3 torch.Size([128, 256])
    #print("f_logvar:", f_logvar.shape) # torch.Size([128, 256])
    #print("f_post:", f_post.shape) # torch.Size([128, 256])
    #print("z_mean_post:", z_mean_post.shape) # torch.Size([128, 8, 32])
    #print("z_logvar_post:", z_logvar_post.shape) # torch.Size([128, 8, 32])
    #print("z_post:", z_post.shape) # torch.Size([128, 8, 32])
    #print("z_mean_prior:", z_mean_prior.shape) # [128, 8, 32]
    #print("z_logvar_prior:", z_logvar_prior.shape) # [128, 8, 32]
    #print("z_prior:", z_prior.shape) # [128, 8, 32]
    # recon_x: [128, 8, 3, 64, 64]
    # pred_area: [1024, 9]

    mi_xs = compute_mi(f, (f_mean, f_logvar))
    n_bs = z_post.shape[0]
    
    mi_xzs = [compute_mi(z_post_t, (z_post_mean_t, z_post_logvar_t)) \
                for z_post_t, z_post_mean_t, z_post_logvar_t in \
                zip(z_post.permute(1,0,2), z_post_mean.permute(1,0,2), z_post_logvar.permute(1,0,2))]
    mi_xz = torch.stack(mi_xzs).sum()

    if opt.loss_recon == 'L2': # True branch
        l_recon = F.mse_loss(recon_x, x, reduction='sum')
    else:
        l_recon = torch.abs(recon_x - x).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1])) # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1])) # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar) # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar) # [128, 8, 32]
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size

    batch_size, n_frame, z_dim = z_post_mean.size()

    con_loss_c = contras_fn(f_mean, f_mean_c)
    con_loss_m = contras_fn(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))

    # calculate the mutual infomation of f and z
    mi_fz = torch.zeros((1)).cuda()
    if True: # 0.1
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # batch_size x batch_size x f_dim
        _logq_f_tmp = log_density(f.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, opt.f_dim), # [8, 128, 1, 256]
                                  f_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim), # [8, 1, 128, 256]
                                  f_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim)) # [8, 1, 128, 256]

        # n_frame x batch_size x batch_size x f_dim
        _logq_z_tmp = log_density(z_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim), # [8, 128, 1, 32]
                                  z_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim), # [8, 1, 128, 32]
                                  z_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim)) # [8, 1, 128, 32]
        #print("_logq_f_tmp:", _logq_f_tmp.shape) # [8, 128, 128, 256]
        #print("_logq_z_tmp:", _logq_z_tmp.shape) # [8, 128, 128, 32]

        _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3) # [8, 128, 128, 288]

        #print(batch_size, opt.dataset_size) # 28, 9000
        logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size)) # [8, 128]
        logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size)) # [8, 128]
        logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size)) # [8, 128]
        # n_frame x batch_size
        mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()

    #loss = l_recon + kld_f*opt.weight_f + kld_z*opt.weight_z + trp_loss*opt.weight_triple + motion_loss*opt.weight_motion + mi_fz*opt.weight_MI
    loss = l_recon + kld_f*opt.weight_f + kld_z*opt.weight_z + mi_fz
    if opt.weight_c_aug:
        loss += con_loss_c*opt.weight_c_aug
    if opt.weight_m_aug:
        loss += con_loss_m*opt.weight_m_aug

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()
    '''if optimizer_cls and mode == "train":
        optimizer_cls.step()'''
    return [i.data.cpu().numpy() for i in [l_recon, kld_f, kld_z, con_loss_c, con_loss_m, mi_fz, mi_xs, mi_xz]]

def test_epoch(epoch, classifier, cdsvae, test_loader, writer):
    cdsvae.eval()
    label1_all, label2_all, label3_all = list(), list(), list()
    pred1_all, pred2_all, pred3_all = list(), list(), list()
    label_gt = list()
    for data in test_loader:
        x, label_A, label_D = reorder(data['images']), data['A_label'], data['D_label']
        x = x.cuda()

        """ #1 change"""
        if opt.type_gt == "action":
            recon_x_sample, recon_x = cdsvae.forward_fixed_motion_for_classification(x)
        else:
            recon_x_sample, recon_x = cdsvae.forward_fixed_content_for_classification(x)

        with torch.no_grad():
            """ #2 change"""
            pred1_1, pred1_2, pred1_3, pred1_4, pred1_5 = classifier(x)
            pred2_1, pred2_2, pred2_3, pred2_4, pred2_5 = classifier(recon_x_sample)
            pred3_1, pred3_2, pred3_3, pred3_4, pred3_5 = classifier(recon_x)
            if opt.type_gt == "action":
                pred1, pred2, pred3 = pred1_1, pred2_1, pred3_1
            elif opt.type_gt == "skin":
                pred1, pred2, pred3 = pred1_2, pred2_2, pred3_2
            elif opt.type_gt == "top":
                pred1, pred2, pred3 = pred1_3, pred2_3, pred3_3
            elif opt.type_gt == "pant":
                pred1, pred2, pred3 = pred1_4, pred2_4, pred3_4
            elif opt.type_gt == "hair":
                pred1, pred2, pred3 = pred1_5, pred2_5, pred3_5

            pred1 = F.softmax(pred1, dim = 1)
            pred2 = F.softmax(pred2, dim = 1)
            pred3 = F.softmax(pred3, dim = 1)

        label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
        label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
        label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)

        pred1_all.append(pred1.detach().cpu().numpy())
        pred2_all.append(pred2.detach().cpu().numpy())
        pred3_all.append(pred3.detach().cpu().numpy())

        """ #3 change"""
        if opt.type_gt == "action":
            label_gt.append(label_D.numpy())
        elif opt.type_gt == "skin":
            label_gt.append(label_A[:,0].numpy())
        elif opt.type_gt == "top":
            label_gt.append(label_A[:,1].numpy())
        elif opt.type_gt == "pant":
            label_gt.append(label_A[:,2].numpy())
        elif opt.type_gt == "hair":
            label_gt.append(label_A[:,3].numpy())

        label1_all.append(label1)
        label2_all.append(label2)
        label3_all.append(label3)
    label1_all = np.hstack(label1_all)
    label2_all = np.hstack(label2_all)
    label3_all = np.hstack(label3_all)
    label_gt = np.hstack(label_gt)

    pred1_all = np.vstack(pred1_all)
    pred2_all = np.vstack(pred2_all)
    pred3_all = np.vstack(pred3_all)

    acc = (label1_all == label2_all).mean()
    kl  = KL_divergence(pred2_all, pred1_all)

    """These scores are influented by label distribution. select pred2_all with uniform label distribution"""
    nSample_per_cls = min([(label_gt==i).sum() for i in np.unique(label_gt)])
    #print(nSample_per_cls)
    index  = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
    pred2_selected = pred2_all[index]

    IS  = inception_score(pred2_selected)
    H_yx = entropy_Hyx(pred2_selected)
    H_y = entropy_Hy(pred2_selected)
    
    print("###############")
    print('Test acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc*100, kl, IS, H_yx, H_y))
    print("###############")

    if writer is not None:
        writer.add_scalar("Test/acc", acc, epoch)
        writer.add_scalar("Test/kl", kl, epoch)
        writer.add_scalar("Test/IS", IS, epoch)
        writer.add_scalar("Test/H_yx", H_yx, epoch)
        writer.add_scalar("Test/H_y", H_y, epoch)


def main(opt):
    name = 'CDSVAE_Sprite_epoch-{}_bs-{}_decoder={}{}x{}-rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
           '-weight:kl_f={}-kl_z={}-c_aug={}-m_aug={}-{}-sche_{}-{}'.format(
               opt.nEpoch, opt.batch_size, opt.decoder, opt.image_width, opt.image_width, opt.rnn_size, opt.g_dim, opt.f_dim, opt.z_dim, opt.lr,
               opt.weight_f, opt.weight_z, opt.weight_c_aug, opt.weight_m_aug,
               opt.loss_recon, opt.sche, opt.note)

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    log = os.path.join(opt.log_dir, 'log.txt')
    mi_path = os.path.join(opt.log_dir, 'mi.txt')

    summary_dir = os.path.join('./summary/', opt.dataset, name)
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    print_log("Random Seed: {}".format(opt.seed), log)
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_dir)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)

    # control the sequence sample
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    print_log('Running parameters:')
    print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam
    cdsvae = CDSVAE(opt)

    cdsvae.apply(utils.init_weights)
    optimizer = opt.optimizer(cdsvae.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.sche == "cosine":
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.nEpoch+1)//2, eta_min=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)
    elif opt.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch//2, gamma=0.5)
    elif opt.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    if opt.model_dir != '':
        cdsvae =  saved_model['cdsvae']

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        cdsvae = nn.DataParallel(cdsvae)

    cdsvae = cdsvae.cuda()
    print_log(cdsvae, log)

    classifier = classifier_Sprite_all(opt)
    opt.cls_path = './judges/Sprite/sprite_judge.tar'
    loaded_dict = torch.load(opt.cls_path)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    N, seq_len, dim1, dim2, n_c = train_data.data.shape
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=opt.batch_size, # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=opt.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    test_video_enumerator = get_batch(test_loader)
    opt.dataset_size = len(train_data)

    epoch_loss = Loss()
    contras_fn = contrastive_loss(tau=0.5, normalize=True)
    # --------- training loop ------------------------------------
    cur_step = 0
    for epoch in range(opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        cdsvae.train()
        epoch_loss.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(max_value=len(train_loader)).start()
        for i, data in enumerate(train_loader):
            '''
            images : torch.Size([128, 8, 64, 64, 3])
            A_label : torch.Size([128, 4])
            D_label : torch.Size([128])
            OF_label : torch.Size([128, 8, 9])
            mask : torch.Size([128, 8, 9])
            images_pos : torch.Size([128, 8, 64, 64, 3])
            images_neg : torch.Size([128, 8, 64, 64, 3])
            index : torch.Size([128])
            '''
            
            progress.update(i+1)
            x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
            x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()

            # train frame_predictor
            recon, kld_f, kld_z, con_loss_c, con_loss_m, mi_fz, mi_xs, mi_xz = train(x, label_A, label_D, c_aug, m_aug, cdsvae, optimizer, contras_fn, opt)

            lr = optimizer.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar("lr", lr, cur_step)
                writer.add_scalar("Train/mse", recon.item(), cur_step)
                writer.add_scalar("Train/kld_f", kld_f.item(), cur_step)
                writer.add_scalar("Train/kld_z", kld_z.item(), cur_step)
                writer.add_scalar("Train/con_loss_c", con_loss_c.item(), cur_step)
                writer.add_scalar("Train/con_loss_m", con_loss_m.item(), cur_step)
                writer.add_scalar("Train/mi_fz", mi_fz.item(), cur_step)
                print_log('train_xs {} {}'.format(cur_step, mi_xs.item()), mi_path, False)
                print_log('train_xz {} {}'.format(cur_step, mi_xz.item()), mi_path, False)
                print_log('train_fz {} {}'.format(cur_step, mi_fz.item()), mi_path, False)
                cur_step += 1

            epoch_loss.update(recon, kld_f, kld_z, con_loss_c, con_loss_m)
            if i % 100 == 0 and i:
                print_log('[%02d] recon: %.3f | kld_f: %.3f | kld_z: %.3f | con_loss_c: %.5f |'
                          ' con_loss_m: %.5f | lr: %.5f' % (epoch, recon, kld_f, kld_z, con_loss_c, con_loss_m, lr), log)

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        print_log('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_loss_c: %.5f |'
                  ' con_loss_m: %.5f | lr: %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2],
                                                    avg_loss[3], avg_loss[4], lr), log)

        if epoch%opt.evl_interval == 0 or epoch == opt.nEpoch-1:
            cdsvae.eval()
            # save the model
            net2save = cdsvae.module if torch.cuda.device_count() > 1 else cdsvae
            torch.save({
                'model': net2save.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (opt.log_dir, epoch))

        if epoch == opt.nEpoch-1 or epoch % 5 == 0:
            val_mse = val_kld_f = val_kld_z = val_c_loss = val_m_loss = val_mi_xs = val_mi_fz = val_mi_xz = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
                x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()

                with torch.no_grad():
                    recon, kld_f, kld_z, con_loss_c, con_loss_m, mi_fz, mi_xs, mi_xz = train(x, label_A, label_D, c_aug, m_aug, cdsvae, optimizer, contras_fn, opt, mode="val")

                val_mse += recon
                val_kld_f += kld_f
                val_kld_z += kld_z
                val_c_loss += con_loss_c
                val_m_loss += con_loss_m
                val_mi_xs += mi_xs
                val_mi_xz += mi_xz
                val_mi_fz += mi_fz

            n_batch = len(test_loader)
            if writer is not None:
                writer.add_scalar("Val/mse", val_mse.item()/n_batch, epoch)
                writer.add_scalar("Val/kld_f", val_kld_f.item()/n_batch, epoch)
                writer.add_scalar("Val/kld_z", val_kld_z.item()/n_batch, epoch)
                writer.add_scalar("Val/con_loss_c", val_c_loss.item()/n_batch, epoch)
                writer.add_scalar("Val/con_loss_m", val_m_loss.item()/n_batch, epoch)
                writer.add_scalar("Val/mi_fz", val_mi_fz.item()/n_batch, epoch)
                print_log('val_xs {} {}'.format(epoch, val_mi_xs.item()/n_batch), mi_path, False)
                print_log('val_xz {} {}'.format(epoch, val_mi_xz.item()/n_batch), mi_path, False)
                print_log('val_fz {} {}'.format(epoch, val_mi_fz.item()/n_batch), mi_path, False)

        if epoch == opt.nEpoch-1 or epoch % 10 == 0:
            test_epoch(epoch, classifier, cdsvae, test_loader, writer)

# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0,1,4,2,3)


def get_batch(train_loader):
    while True:
        for sequence in train_loader:
            yield sequence


def print_log(print_string, log=None, verbose=True):
  if verbose:
      print("{}".format(print_string))
  if log is not None:
      log = open(log, 'a')
      log.write('{}\n'.format(print_string))
      log.close()

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


