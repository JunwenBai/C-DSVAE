import math
import torch
import socket
import argparse
import os
import numpy as np
import random

import scipy.misc
import matplotlib
matplotlib.use('agg')
from PIL import Image, ImageDraw

from torch.autograd import Variable
from torchvision import datasets, transforms
import imageio
from dataloader.sprite import Sprite

import pickle
hostname = socket.gethostname()


def load_dataset(opt):
    if opt.dataset == 'Sprite':
        data = pickle.load(open('../dataset/Sprite/data.pkl', 'rb'))
        X_train, X_test, A_train, A_test = data['X_train'], data['X_test'], data['A_train'], data['A_test']
        D_train, D_test = data['D_train'], data['D_test']
        c_augs_train, c_augs_test = data['c_augs_train'], data['c_augs_test']
        m_augs_train, m_augs_test = data['m_augs_train'], data['m_augs_test']

        print("finish loading!")

        train_data = Sprite(train=True, data = X_train, A_label = A_train,
                            D_label = D_train, c_aug = c_augs_train, m_aug = m_augs_train)
        test_data = Sprite(train=False, data = X_test, A_label = A_test, 
                            D_label = D_test, c_aug = c_augs_test, m_aug = m_augs_test)

    return train_data, test_data


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


import torch.nn as nn
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

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

