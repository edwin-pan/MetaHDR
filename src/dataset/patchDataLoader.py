#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:20:02 2021

@author: edwinpan
"""
import os
import numpy as np
from scipy.interpolate import interp1d
import pickle
import cv2
import logging
from skimage import color

from torch.utils.data import Dataset

np.random.seed(0)

# --- crf_list

def _get_crf_list():
    with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
    crf_list = np.float32([ele.split() for ele in crf_list])
    np.random.RandomState(730).shuffle(crf_list)
    test_crf_list = crf_list[-10:]
    train_crf_list = crf_list[:-10]

    return test_crf_list, train_crf_list


# test_crf_list, train_crf_list = _get_crf_list()


# --- invcrf_list

def _inverse_rf(
        _rf,  # [s]
):
    rf = _rf.copy()
    s, = rf.shape
    rf[0] = 0.0
    rf[-1] = 1.0
    return interp1d(
        rf,
        np.linspace(0.0, 1.0, num=s),
    )(np.linspace(0.0, 1.0, num=s))

# _get_invcrf_list = lambda crf_list: np.array([_inverse_rf(crf) for crf in crf_list])
# test_invcrf_list = _get_invcrf_list(test_crf_list)
# train_invcrf_list = _get_invcrf_list(train_crf_list)


class MemDataset(Dataset):

    def __init__(self, dataset):
        self._arr = []
        for idx, ele in enumerate(dataset):
            logging.info('load dataset[%d]' % idx)
            self._arr.append(ele)
        return

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)

# --- PatchHDRDataset

def _load_pkl(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


# i_dataset_train_posfix_list = _load_pkl('i_dataset_train')
# i_dataset_test_posfix_list = _load_pkl('i_dataset_test')
# # a_dataset_test_posfix_list = _load_pkl('a_dataset_test')


class HDRDataset(Dataset):

    def __init__(self, hdr_prefix, hdr_posfix_list, is_training):
        self._hdr_prefix = hdr_prefix
        self._hdr_posfix_list = hdr_posfix_list
        self.is_training = is_training
        return

    def __getitem__(self, idx):
        return HDRDataset._hdr_read_resize(os.path.join(self._hdr_prefix, self._hdr_posfix_list[idx]), self.is_training)

    def __len__(self):
        return len(self._hdr_posfix_list)

    @staticmethod
    def _hdr_read(path):
        hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    @staticmethod
    def _hdr_resize(img, h, w):
        img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        return img

    @staticmethod
    def _hdr_read_resize(path, is_training):
        hdr = HDRDataset._hdr_read(path)
        h, w, _, = hdr.shape
        ratio = max(512 / h, 512 / w)
        h = round(h * ratio)
        w = round(w * ratio)
        hdr = HDRDataset._hdr_resize(hdr, h, w)

        return hdr
    

class PatchHDRDataset(Dataset):
    def __init__(self, hdr_prefix, hdr_posfix_list, is_training=True, n_way=7, load_to_mem=False):
        self._hdr_dataset = HDRDataset(hdr_prefix, hdr_posfix_list, is_training)
        self.n_way = n_way

        crf_list = self.mem_get_crf_list('/home/users/edwinpan/MetaHDR/dorfCurves.txt')

        self.train_crf_list = crf_list[0]
        self.test_crf_list = crf_list[1]


        get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
        self.test_t_list = get_t_list(7)
        self.train_t_list = get_t_list(600)

        if load_to_mem:
            self._hdr_dataset = MemDataset(self._hdr_dataset)
        self._is_training = is_training
        return

    def mem_get_crf_list(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
        crf_list = np.float32([ele.split() for ele in crf_list])
        np.random.RandomState(730).shuffle(crf_list)
        test_crf_list = crf_list[-10:]
        train_crf_list = crf_list[:-10]

        return test_crf_list, train_crf_list

    def __getitem__(self, idx):
        print("[DEBUG] we are in __getitem__")
        hdr = self._hdr_dataset[idx // 2]
        h, w, _, = hdr.shape
        if h > w:
            hdr = hdr[:512, :, :] if idx % 2 == 0 else hdr[-512:, :, :]
        else:
            hdr = hdr[:, :512, :] if idx % 2 == 0 else hdr[:, -512:, :]
        hdr = PatchHDRDataset._pre_hdr_p2(hdr)
        if self._is_training:
            print("[DEBUG] we are training")
            scale = np.random.uniform(0.5, 2.0)
            hdr = cv2.resize(hdr, (np.round(512 * scale).astype(np.int32), np.round(512 * scale).astype(np.int32)), cv2.INTER_AREA)


            def randomCrop(img, width, height):
                assert img.shape[0] >= height
                assert img.shape[1] >= width
                if img.shape[1] == width or img.shape[0] == height:
                    return img
                x = np.random.randint(0, img.shape[1] - width)
                y = np.random.randint(0, img.shape[0] - height)
                img = img[y:y + height, x:x + width]
                return img

            hdr = randomCrop(hdr, 256, 256)

            hdr = np.rot90(hdr, np.random.randint(4))

            _rand_f_h = lambda: np.random.choice([True, False])
            if _rand_f_h():
                hdr = np.flip(hdr, 0)

            _rand_f_v = lambda: np.random.choice([True, False])
            if _rand_f_v():
                hdr = np.flip(hdr, 1)

            print("[DEBUG] we are about to do exposures")
            # Convert to SETS of LDR -> sample n randomly chosen crfs, without replacement
            n = self.n_way+1
            # Sample n exposure levels
            chosen_exp = np.random.choice(self.train_t_list, n, replace=False).reshape(n,1,1,1)
            gt_imgs = np.repeat(hdr[np.newaxis, ...], n, axis=0)
            sim_exp_imgs = gt_imgs*chosen_exp
            
            clipped_hdr = np.clip(sim_exp_imgs, 0, 1)
            
            chosen_int = np.random.randint(0, self.train_crf_list.shape[1]-1)
            print("[DEBUG] apply crf")
            ldr = apply_rf(clipped_hdr, self.train_crf_list[chosen_int])
            
            ldr_q = np.round(ldr * 255.0).astype(np.uint8)
            
            print("[DEBUG] check")
            # Check to make sure the exposures aren't "illegal"
            upperThresh = 249
            lowerThresh = 6
            limit = 256*256*0.5
            grayscale = color.rgb2gray(ldr_q)*255.0
            
            over_exp = np.greater_equal(grayscale, upperThresh)
            over_count = np.sum(over_exp, axis=(1,2))
            over_bools = over_count > limit
            
            under_exp = np.less_equal(grayscale, lowerThresh)
            under_count = np.sum(under_exp, axis=(1,2))
            under_bools = under_count > limit
            
            score = under_count + over_count
            best_of_the_worst = np.argsort(score)[-2:]
            
            the_bad_ones = under_bools | over_bools
            
            if np.sum(the_bad_ones) >= n-1:
                ldr_tasks = ldr_q[best_of_the_worst]
            else:
                ldr_tasks = np.delete(ldr_q, np.where([the_bad_ones]), axis=0)
                
            num_in_batch = ldr_tasks.shape[0]
            train_ldrs = ldr_tasks[:num_in_batch-1]
            train_hdrs = gt_imgs[:num_in_batch-1]
            train = np.stack([train_ldrs, train_hdrs])
            
            test_ldrs = ldr_tasks[num_in_batch-1]
            test_hdrs = gt_imgs[num_in_batch-1]
            test = np.stack([test_ldrs, test_hdrs])
            test = np.expand_dims(test, axis=1)

            print("[DEBUG] end")
        return train, test

    def __len__(self):
        return 2 * len(self._hdr_dataset)

    @staticmethod
    def _hdr_rand_flip(hdr):
        _rand_t_f = lambda: np.random.choice([True, False])
        if _rand_t_f():
            hdr = np.flip(hdr, 0)
        if _rand_t_f():
            hdr = np.flip(hdr, 1)
        return hdr

    @staticmethod
    def _pre_hdr_p2(hdr):
        hdr_mean = np.mean(hdr)
        hdr = 0.5 * hdr / (hdr_mean + 1e-6)
        return hdr


def apply_rf(img, rf):
    n, h, w, c = img.shape
    k = rf.shape[0]
    interpolator = interp1d(np.arange(k), rf)
    
    return interpolator(img.flatten()*(k-1)).reshape((n, h, w, c))


if __name__ == '__main__':    
    hdr_prefix = '/Users/edwinpan/research/SIGGRAPH_ASIA_2021/SingleHDR_training_data/HDR-Synth'
    dl = PatchHDRDataset(hdr_prefix, i_dataset_train_posfix_list, True)
    dl_iter = iter(dl)
    
    train, test = next(dl_iter)
    
    # hdr_img = next(dl_iter)
    
    # # Convert to SETS of LDR -> sample n randomly chosen crfs, without replacement
    # n = 5+1
    # # Sample n exposure levels
    # chosen_exp = np.random.choice(train_t_list, n, replace=False).reshape(n,1,1,1)
    # gt_imgs = np.repeat(hdr_img[np.newaxis, ...], n, axis=0)
    # sim_exp_imgs = gt_imgs*chosen_exp
    
    # clipped_hdr = np.clip(sim_exp_imgs, 0, 1)
    
    # chosen_int = np.random.randint(0, train_crf_list.shape[1]-1)
    # ldr = apply_rf(clipped_hdr, train_crf_list[chosen_int])
    
    # ldr_q = np.round(ldr * 255.0).astype(np.uint8)
    
    # # Check to make sure the exposures aren't "illegal"
    # upperThresh = 249
    # lowerThresh = 6
    # limit = 256*256*0.5
    # grayscale = color.rgb2gray(ldr_q)*255.0
    
    # over_exp = np.greater_equal(grayscale, upperThresh)
    # over_count = np.sum(over_exp, axis=(1,2))
    # over_bools = over_count > limit
    
    # under_exp = np.less_equal(grayscale, lowerThresh)
    # under_count = np.sum(under_exp, axis=(1,2))
    # under_bools = under_count > limit
    
    # score = under_count + over_count
    # best_of_the_worst = np.argsort(score)[-2:]
    
    # the_bad_ones = under_bools | over_bools
    
    # if np.sum(the_bad_ones) >= n-1:
    #     ldr_tasks = ldr_q[best_of_the_worst]
    # else:
    #     ldr_tasks = np.delete(ldr_q, np.where([the_bad_ones]), axis=0)
        
    # num_in_batch = ldr_tasks.shape[0]
    # train_ldrs = ldr_tasks[:num_in_batch-1]
    # train_hdrs = gt_imgs[:num_in_batch-1]
    # train = np.stack([train_ldrs, train_hdrs])
    
    # test_ldrs = ldr_tasks[num_in_batch-1]
    # test_hdrs = gt_imgs[num_in_batch-1]
    # test = np.stack([test_ldrs, test_hdrs])
    # test = np.expand_dims(test, axis=1)


# import matplotlib.pyplot as plt
# for i in range(n):
#     plt.figure()
#     plt.imshow(grayscale[i])

# Construct meta-tasks


