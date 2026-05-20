import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#推理工具类
def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUDT':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'NUAA':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'IRSTD':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    else:
        with open(dataset_dir + '/' + dataset_name + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        if os.path.exists(dataset_dir + '/' + dataset_name + '/img_idx/test_' + dataset_name + '.txt'):
            with open(dataset_dir + '/' + dataset_name + '/img_idx/test_' + dataset_name + '.txt', 'r') as f:
                test_list = f.read().splitlines()
        else:
            test_list = []
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
        print(dataset_name + '\t' + str(img_norm_cfg))
    return img_norm_cfg


def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'])

    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'],
                                                         gamma=scheduler_settings['gamma'])
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'],
                                                               eta_min=scheduler_settings['min_lr'])

    return optimizer, scheduler


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img        
