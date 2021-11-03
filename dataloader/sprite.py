import random
import os
import numpy as np
import socket
import torch

class Sprite(object):
    def __init__(self, train, data, A_label, D_label, c_aug, m_aug):
        self.data = data
        self.A_label = A_label
        self.D_label = D_label
        self.N = self.data.shape[0]
        self.c_aug = c_aug
        self.m_aug = m_aug
        self.aug_num = c_aug.shape[1]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_ancher = self.data[index] 
        A_label_ancher = self.A_label[index] 
        D_label_ancher = self.D_label[index]
        idx = np.random.randint(self.aug_num)
        c_aug_anchor = self.c_aug[index][idx]
        m_aug_anchor = self.m_aug[index][idx]

        return {"images": data_ancher, "c_aug": c_aug_anchor, "m_aug": m_aug_anchor, "A_label": A_label_ancher, "D_label": D_label_ancher, "index": index}

