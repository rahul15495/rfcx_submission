import torch
import random
import load_mel
import numpy as np
import pandas as pd
from mixup import MixUp
from augment import do_aug
from torch.utils.data import (Dataset , DataLoader , RandomSampler)
from preprocessor import Preprocessor
from tqdm.notebook import tqdm


class BaseDataset(Dataset):
    def __init__(self, root_dir, df,cache=None, enable_mixup=False, enable_aug=False, SAMPLE_RATE=32000, num_classes=24):
        
        self.root_dir = root_dir
        self.df= df
        self.all_audio= None
        self.enable_mixup = enable_mixup
        self.enable_aug = enable_aug
        self.SAMPLE_RATE = SAMPLE_RATE
        self.num_classes= num_classes
        self.cache = cache
        self.num_samples= len(self.df)
        self.mixup= MixUp(load_mel.denoise, SAMPLE_RATE)
        self.max_mixup = 4
        self.load_all_audio()

        
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
    
    def load_all_audio(self):
        
        if self.cache:
            print('using cached dataset')
            self.all_audio= self.cache
            
        else:
            
            all_audio= {}
            for recording_id in tqdm(self.df.recording_id.unique()):
                all_audio[recording_id] = self.load_audio(recording_id)

            self.all_audio = all_audio
        print('all audio loaded')
        
    """
    def do_mixup(self, idx):
        
        other_idx = self.get_random_idx(idx)
        
        y1, label1 = self.get_sample_audio_label(idx)
        y2, label2 = self.get_sample_audio_label(other_idx)
        
        _,y= self.mixup(y1, y2)
        label= np.bitwise_or(label1, label2)

        return y, label
    """
    
    def do_mixup(self, idx):

        y1, label1 = self.get_sample_audio_label(idx)    
        
        for e in range(0,random.randint(1, self.max_mixup)):
            
            other_idx = self.get_random_idx(idx)
            y2, label2 = self.get_sample_audio_label(other_idx)

            _,y1= self.mixup(y1, y2)
            
            label1= np.bitwise_or(label1, label2)

        return y1, label1
    
    def get_sample_audio_label(self, idx):
        
        row=self.df.loc[idx]
        recording_id= row.recording_id

        y = self.all_audio[recording_id]

        begin= np.ceil(row.t_min * self.SAMPLE_RATE).astype(int)
        end= np.ceil(row.t_max * self.SAMPLE_RATE).astype(int)
        wave = y[begin:end].copy()

        #repeat samples to match 10 sec audio splits
        out_wave = np.resize(wave, self.SAMPLE_RATE*10)
        
        
        label= np.zeros(self.num_classes, np.int)
        label[row['species_id']-1] =1
        
        return out_wave ,label
    
    def __getitem__(self, idx):
        
        if self.enable_mixup and (random.random() >0.5):
            #print('mixup')
            
            try:
                y,label = self.do_mixup(idx)
            except:
                y, label = self.get_sample_audio_label(idx)
        else:
            y, label = self.get_sample_audio_label(idx)

        
        if self.enable_aug and (random.random() >0.5):
            #print('aug')
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
                                 batch_size=1,
                                 sampler= RandomSampler(test_dataset),
                                 collate_fn =collate_fn,
                                 num_workers= 4,
                                 prefetch_factor=4,
                                )

    print(f'training: number of docs : {len(train_dataloader)}')
    print(f'evaluation: number of docs : {len(eval_dataloader)}')
    
    return train_dataloader, eval_dataloader
    