import torch
import random
import load_mel
import numpy as np
import pandas as pd
from mixup import MixUp
from augment import do_aug
from torch.utils.data import (Dataset , DataLoader , RandomSampler)
from preprocessor import Preprocessor


def get_train_mapping(df, num_classes=24):
    recording_wise_labels= []

    for idx, _id in enumerate(df['recording_id'].unique()):

        temp= np.zeros(num_classes, np.int)

        for _class in df.loc[df['recording_id'] == _id]['species_id'].unique():
            temp[_class-1]=1
            
        recording_wise_labels.append({'recording_id':_id, 'labels':temp})
        
    return recording_wise_labels


class BaseDataset(Dataset):
    def __init__(self, root_dir, mapping, enable_mixup=False, enable_aug=False, SAMPLE_RATE=16000):
        
        self.root_dir = root_dir
        self.mapping= mapping
        self.enable_mixup = enable_mixup
        self.enable_aug = enable_aug
        self.SAMPLE_RATE = SAMPLE_RATE
        self.num_samples= len(self.mapping)
        self.mixup= MixUp(load_mel.denoise, SAMPLE_RATE)

        
    def __len__(self):
        return self.num_samples
    
    def get_random_idx(self, curr):
        other= random.choice(range(0, self.num_samples))
        if other==curr:
            other = self.get_random_idx(curr)
        return other
    
    def load_audio(self, recording_id):
        input_file = f'{self.root_dir}/{recording_id}.flac'
        y,_= load_mel.load_audio(input_file, self.SAMPLE_RATE)
        return y
    
    def do_mixup(self, idx):
        
        other_idx = self.get_random_idx(idx)
        
        record1 = self.mapping[idx]
        record2 = self.mapping[other_idx]
        y1, label1 = self.load_audio(record1['recording_id']) , record1['labels']
        y2, label2 = self.load_audio(record2['recording_id']) , record2['labels']
        
        _,y= self.mixup(y1, y2)
        label= np.bitwise_or(label1, label2)

        return y, label
    
    def __getitem__(self, idx):
        
        if self.enable_mixup and (random.random() >0.5):
            
            try:
                y,label = self.do_mixup(idx)
            except:
                record= self.mapping[idx]
                y= self.load_audio(record['recording_id'])
                label= record['labels']                
        else:
            record= self.mapping[idx]
            y= self.load_audio(record['recording_id'])
            label= record['labels']

        
        if self.enable_aug and (random.random() >0.5):
            y = do_aug(y, self.SAMPLE_RATE)
            
        feat= load_mel.get_spectrogram(y,self.SAMPLE_RATE,apply_denoise=False,return_audio=False)
        return y, feat, label
    


def get_dataloaders(train_dataset, test_dataset, batch_size, device):
    
    preprocessor= Preprocessor(hidden_size =768, dr=1, device=device)

    def collate_fn(data):

        specs, pos_encs, attn_masks, labels  = [] , [] , [] , []

        for item in data:
            _, spec, label= item

            spec = torch.tensor(spec)
            spec = spec.permute(1, 0)

            label = torch.FloatTensor(label).unsqueeze(0)

            spec, pos_enc, attn_mask = preprocessor.process_MAM_data(spec=spec)

            specs.append(spec)
            pos_encs.append(pos_enc)
            attn_masks.append(attn_mask)
            labels.append(label)

        *batch_stacked, labels_stacked  = [torch.cat(e,0).to(device=device) for e in [specs, pos_encs, attn_masks, labels]]

        return tuple(batch_stacked), labels_stacked

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler= RandomSampler(train_dataset),
                                  collate_fn =collate_fn,
                                  num_workers= 4,
                                  prefetch_factor=4,
                                 )

    eval_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 sampler= RandomSampler(test_dataset),
                                 collate_fn =collate_fn,
                                 num_workers= 4,
                                 prefetch_factor=4,
                                )

    print(f'training: number of docs : {len(train_dataloader)}')
    print(f'evaluation: number of docs : {len(eval_dataloader)}')
    
    return train_dataloader, eval_dataloader
    