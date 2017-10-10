from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DataLoader():
    
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = self.opt.seq_per_img

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file for train set: ', opt.input_label_h5_train, opt.input_image_h5_train)
        print('DataLoader loading h5 file for val set: ', opt.input_label_h5_val, opt.input_image_h5_val)
        self.h5_label_file_train = h5py.File(self.opt.input_label_h5_train)
        self.h5_image_file_train = h5py.File(self.opt.input_image_h5_train)
        self.h5_label_file_val = h5py.File(self.opt.input_label_h5_val)
        self.h5_image_file_val = h5py.File(self.opt.input_image_h5_val)


        # extract image size from dataset
        images_size = self.h5_image_file_train['images'].shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images_train = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' %(self.num_images, 
                    self.num_channels, self.max_image_size, self.max_image_size))
        
        images_size = self.h5_image_file_val['images'].shape
        self.num_images_val = images_size[0]

        # load in the sequence data
        seq_size = self.h5_label_file_train['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix_train = self.h5_label_file_train['label_start_ix'][:]
        self.label_end_ix_train = self.h5_label_file_train['label_end_ix'][:]
        self.label_start_ix_val = self.h5_label_file_val['label_start_ix'][:]
        self.label_end_ix_val = self.h5_label_file_val['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        '''
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))
        '''
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size

        img_batch = np.ndarray([batch_size, 3, 256,256], dtype = 'float32')
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'float32')
        if split == 'train' :
            max_index = self.num_images_train
        else if split == 'val' :
            max_index = self.num_images_val
        else :
            print "ERROR no such split!!!"
        wrapped = False

        #infos = []

        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = ri

            # fetch image
            #img = self.load_image(self.image_info[ix]['filename'])
            if split == 'train' :
                img = self.h5_image_file_train['images'][ix, :, :, :]
            else if split == 'val' :
                img = self.h5_image_file_val['images'][ix, :, :, :]
            img_batch[i] = preprocess(torch.from_numpy(img.astype('float32')/255.0)).numpy()

            # fetch the sequence labels
            if split == 'train' :
                ix1 = self.label_start_ix_train[ix] - 1 #label_start_ix starts from 1
                ix2 = self.label_end_ix_train[ix] - 1
            else if split == 'val' :
                ix1 = self.label_start_ix_val[ix] - 1 #label_start_ix starts from 1
                ix2 = self.label_end_ix_val[ix] - 1

            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                for q in range(self.seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    if split == 'train' :
                        seq[q, :] = self.h5_label_file_train['labels'][ixl, :self.seq_length]
                    elif split == 'val' :
                        seq[q, :] = self.h5_label_file_val['labels'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                if split == 'train' :
                    seq = self.h5_label_file_train['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]
                elif split == 'val' :
                    seq = self.h5_label_file_val['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]

            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq

            # record associated info as well
            '''
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            '''

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['images'] = img_batch
        data['labels'] = label_batch
        data['masks'] = mask_batch 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        #data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0
        
