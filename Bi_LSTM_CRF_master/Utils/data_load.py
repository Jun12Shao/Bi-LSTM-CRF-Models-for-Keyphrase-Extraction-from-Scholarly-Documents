# -*- coding:utf-8 -*-

# -*- coding:utf-8 -*-
import torch
import torch.utils.data as data
import os
import numpy as np
import pickle



def DatasetLoader(args, mode='train'):
    ## Create_dataset
    ## load data buckets and do the transformation
    if args.mode == 'train':
        data_path = os.path.join(args.root, args.train_file)
    elif args.mode == 'valid':
        data_path = os.path.join(args.root, args.valid_file)
    else:
        data_path = os.path.join(args.root, args.test_file)
    ##  corpus
    with open(data_path, 'rb') as f:
        buckets, masks = pickle.load(f, encoding='latin1')
        f.close()


    if args.dataset_name == 'KP':
        ## DataLoader
        dataloaders = []
        k=0
        for i in range(len(buckets)):
            for n in range(len(buckets[i])):
                if buckets[i][n].shape[1]<masks[i][n]+2:
                    masks[i][n]=buckets[i][n].shape[1]-2
                    k+=1
        print("Number of numatch:", k)


        for i in range(len(buckets)):
            dataset = KPDataset(args, mode, buckets[i], masks[i])
            dataloader = data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True)
            dataloaders.append(dataloader)

    else:
        raise ValueError("Dataset [%s] not recognized." % args.dataset_name)


    return dataloaders



class KPDataset(data.Dataset):
    def __init__(self, args, mode,bucket,mask):
        super(KPDataset, self).__init__()
        self._name = 'KPDataset'
        self.args=args
        self.mode=mode
        self.root=args.root
        self.bucket=bucket
        self.mask=mask
        self.dataset_size=len(bucket)


    @property
    def name(self):
        return self._name


    @property
    def path(self):
        return self.root

    def __getitem__(self, index):
        assert (index < self.dataset_size)

        return  torch.tensor(self.bucket[index],dtype=torch.long),torch.tensor(self.mask[index],dtype=torch.long)


            #    torch.tensor(self.data_set[2][index],dtype=torch.long),torch.tensor(self.data_set[3][index],dtype=torch.long)

    def __len__(self):
        return self.dataset_size

