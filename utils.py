import math
import torch
import socket
import argparse
import os
import numpy as np
import random

from dataloader.sprite import Sprite
import pickle

def load_dataset(opt):
    if opt.dataset == 'Sprite':
        import pickle
        data = pickle.load(open("../dataset/Sprite/data.pkl", "rb"))
        X_train, X_test, A_train, A_test = data['X_train'], data['X_test'], data['A_train'], data['A_test']
        D_train, D_test = data['D_train'], data['D_test']
        c_augs_train, c_augs_test = data['c_augs_train'], data['c_augs_test']
        m_augs_train, m_augs_test = data['m_augs_train'], data['m_augs_test']
        
        print("finish loading!")

        train_data = Sprite(train=True, data = X_train, A_label = A_train,
                            D_label = D_train, c_aug = c_augs_train, m_aug = m_augs_train)
        test_data = Sprite(train=False, data = X_test, A_label = A_test, 
                            D_label = D_test, c_aug = c_augs_test, m_aug = m_augs_test)
    else:
        raise ValueError('unknown dataset')

    return train_data, test_data


def clear_progressbar():
    print("\033[2A")
    print("\033[2K")
    print("\033[2A")

