import sys
sys.path.append('../s3prl')

import yaml
import torch
import pandas as pd
from argparse import Namespace
from model_builder import Wrapper_Model
from metrics import LWLRAP
from dataloader import (get_train_mapping, BaseDataset , get_dataloaders)
from model import TransformerModel , TransformerForMaskedAcousticModel , TransformerConfig

class Runner():
    def __init__(self, device, args):
        self.device = device
        self.base_transformer_model = None
        self.model = None
        self.train_dataloader= None
        self.eval_dataloader= None
        self.args = args
        
    def set_transformer_model(self):
        '''
        This Function loads the base transformer model.
        
        Args:
            transformer_config_path : config path(yaml) of the transformer
            transformer_weights_path : optional . if given loads the weight as well
        
        Returns:None
        '''

        # load base transformer model from config
        with open(self.args.transformer_config_path, 'r') as file:
            config= yaml.load(file, yaml.FullLoader)        

        model_config = TransformerConfig(config)
        input_dim = config['transformer']['input_dim']
        
        dr= model_config.downsample_rate
        hidden_size = model_config.hidden_size
        output_attention= False
        
        base_transformer_model = TransformerModel(model_config,input_dim,output_attentions=output_attention).to('cpu')

        #load weights
        if self.args.transformer_weights_path:
            ckpt = torch.load(self.args.transformer_weights_path, map_location='cpu')
            base_transformer_model.load_state_dict(ckpt['Transformer'])

        self.base_transformer_model = base_transformer_model
        
    def set_model(self,transformer_weights_path=None, ckpt_path=None):
        self.set_transformer_model()
        self.model = Wrapper_Model(self.base_transformer_model)
        
        if self.args.ckpt_path:
            ckpt = torch.load(self.args.ckpt_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            
        if not self.args.training['upstream']:
            self.model.upstream.eval()
            
        self.model.to(self.device)
        
    def set_data_loader(self):
        
        df= pd.read_csv(self.args.csv_path)
        
        mapping = get_train_mapping(df)
        train_dataset = BaseDataset(self.args.data_dir, mapping,  enable_mixup=True, enable_aug=False)
        test_dataset = BaseDataset(self.args.data_dir, mapping,  enable_mixup=False, enable_aug=False)
        train_dataloader, eval_dataloader = get_dataloaders(train_dataset, test_dataset, self.args.batch_size,self.device)
        self.train_dataloader= train_dataloader
        self.eval_dataloader= eval_dataloader        
        