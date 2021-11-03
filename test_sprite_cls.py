import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import nn.functional as F
import argparse
import os
import json
from model import CDSVAE, classifier_Sprite_all
import utils
import numpy as np

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

parser.add_argument('--model_epoch', type=int, default=200, help='ckpt epoch')
parser.add_argument('--model_dir', default='', help='ckpt directory')
parser.add_argument('--type_gt',  type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--niter', type=int, default=300, help='number of runs for testing')

opt = parser.parse_args()

def reorder(sequence):
    return sequence.permute(0,1,4,2,3)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


def main(opt):
    if opt.model_dir != '':
        saved_model = torch.load('%s/model%d.pth' % (opt.model_dir, opt.model_epoch))
        model_dir = opt.model_dir
        opt.model_dir = model_dir
    else:
        raise ValueError('missing checkpoint')

    log = os.path.join(opt.log_dir, 'log.txt')
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
    dtype = torch.cuda.FloatTensor

    print_log('Running parameters:')
    print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    if opt.model_dir != '':
        cdsvae = CDSVAE(opt)
        cdsvae.load_state_dict(saved_model['model'])

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        cdsvae = nn.DataParallel(cdsvae)
    cdsvae = cdsvae.cuda()
    print_log(cdsvae, log)

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    opt.g_dim = 128
    opt.rnn_size = 256
    classifier = classifier_Sprite_all(opt)
    opt.resume = './judges/Sprite/sprite_judge.tar'
    loaded_dict = torch.load(opt.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()

    # --------- training loop ------------------------------------
    for epoch in range(opt.niter):

        print("Epoch", epoch)
        cdsvae.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
            x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()

            if opt.type_gt == "action":
                recon_x_sample, recon_x = cdsvae.forward_fixed_motion_for_classification(x)
            else:
                recon_x_sample, recon_x = cdsvae.forward_fixed_content_for_classification(x)
            
            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

                pred1 = F.softmax(pred_action1, dim = 1)
                pred2 = F.softmax(pred_action2, dim = 1)
                pred3 = F.softmax(pred_action3, dim = 1)
            
            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
            label2_all.append(label2)
            
            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(label_D.detach().cpu().numpy())

            def count_D(pred, label, mode=1):
                return (pred//mode) == (label//mode)

            acc0_sample = count_D(np.argmax(pred_action2.detach().cpu().numpy(), axis=1), label_D.cpu().numpy()).mean()
            acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1) == label_A[:, 0].cpu().numpy()).mean()
            acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1) == label_A[:, 1].cpu().numpy()).mean()
            acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1) ==  label_A[:, 2].cpu().numpy()).mean()
            acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1) == label_A[:, 3].cpu().numpy()).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample
            
        print('Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                                                       mean_acc0_sample / len(test_loader)*100,
                                                       mean_acc1_sample / len(test_loader)*100, mean_acc2_sample / len(test_loader)*100,
                                                       mean_acc3_sample / len(test_loader)*100, mean_acc4_sample / len(test_loader)*100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)

        acc = (label_gt == label2_all).mean()
        kl  = KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt==i).sum() for i in np.unique(label_gt)])
        index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        IS  = inception_score(pred2_selected)
        H_yx = entropy_Hyx(pred2_selected)
        H_y = entropy_Hy(pred2_selected)

        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc*100, kl, IS, H_yx, H_y))


def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h

def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis = 1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h

def inception_score(p_yx,  eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d

def print_log(print_string, log=None):
    print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()

if __name__ == '__main__':
    main(opt)
