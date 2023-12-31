#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:26:34 2023

@author: abhishek
"""

import os
import numpy as np
import scipy.io as sio
from customdataset2 import Customdataset
import torch
from torch.utils.data import DataLoader

__all__ = ['Customdataloader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class Customdataloader(object):
    r""" PyTorch DataLoader for custom dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario):
        #assert os.path.isdir(root)
        assert scenario in {"in", "out"}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_train = os.path.join(root,"val/dl")
        dir_val = os.path.join(root, "val/dl")
        dir_test = os.path.join(root,"val/dl")
        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # Training data loading
        #data_train = sio.loadmat(dir_train)['HT']
        #data_train = torch.tensor(data_train, dtype=torch.float32).view(
         #   data_train.shape[0], channel, nt, nc)
        train_dataset = Customdataset(dir_train,'HT')
        self.train_dataset = train_dataset

        # Validation data loading
        val_dataset = Customdataset(dir_val,'HT')
        self.val_dataset = val_dataset

        # Test data loading, including the sparse data and the raw data
        test_dataset = Customdataset(dir_test,'HT')
        self.test_dataset = test_dataset


    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
       # if self.pin_memory is True:
        #    train_loader = PreFetcher(train_loader)
        #    val_loader = PreFetcher(val_loader)
        #    test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader
