s3prl_path= '../s3prl'
import sys
sys.path.append(f'{s3prl_path}/transformer/')
sys.path.append(f'{s3prl_path}/')

import torch
from torch import nn
from model import TransformerModel , TransformerForMaskedAcousticModel , TransformerConfig
import transformer


class Upstream_Model(torch.nn.Module):
  
    def __init__(self, base_transformer_model: TransformerForMaskedAcousticModel):
        super(Upstream_Model,self).__init__()
        self.transformer = base_transformer_model
        self.maxlen=3000
    
    def split(self,inp):
        #shape of each input is (batch_size, sequence, mel_features)
        #goal is to split the sequence if the sequence length is greater tha maxlen
        sequence_length = inp.shape[1]
        axes_length= len(inp.shape)
        
        if sequence_length> self.maxlen:
            
            split_size = list(range(self.maxlen,sequence_length, self.maxlen))
            rem = sequence_length % self.maxlen
            if rem:
                split_size += [rem]

            return torch.split(inp, split_size, dim=1)
            
        else:
            return [inp]
            
        
    def forward(self, spec, pos_enc, attn_mask):
                
        split_spec= self.split(spec)
        split_pos_enc= self.split(pos_enc)
        split_attn_mask= self.split(attn_mask)
        
        layer1 ,layer2 ,layer3= [], [] ,[]
        
        for a,b,c in zip(split_spec, split_pos_enc, split_attn_mask) :
            
            _layer1, _layer2, _layer3 = self.transformer(spec_input=a,
                                        pos_enc=b,
                                        attention_mask=c,
                                        output_all_encoded_layers=True)
            
            layer1.append(_layer1)
            layer2.append(_layer2)
            layer3.append(_layer3)
            
            
        stacked_encoded_layers= torch.stack([torch.cat(layer, axis=1) for layer in [layer1, layer2, layer3]], dim=-1) # (B,N_layers, T,D)
        return stacked_encoded_layers
    

class Downstream_Model(nn.Module):
    '''
    Takes input shape of (Batch_Size, Encoded_layers, Time_step , Hidden_Size_Of_Upstream)
    for eg: (1,4801, 768,3)
    '''
    
    def __init__(self,encoder_num_layers=3, input_dim=768, hidden_dim=128, class_num=24):
        super(Downstream_Model, self).__init__()
        
        self.num_layers  = encoder_num_layers
        self.weight      = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, dropout=0.3,
                          batch_first=True, bidirectional=False)

        self.out = nn.Linear(hidden_dim, class_num)
        self.out_fn = nn.Sigmoid()

    def forward(self, x):
        #weighted_sum. expected_output (Batch_Size, Time_step , Hidden_Size_Of_Upstream)
        softmax_weight = nn.functional.softmax(self.weight, dim=-1)
        x_weighted_sum = torch.matmul(x,softmax_weight)
        
        _, h_n = self.rnn(x_weighted_sum)
        hidden = h_n[-1, :, :]
        logits = self.out(hidden)
        result = self.out_fn(logits)
        return result
    
class Wrapper_Model(torch.nn.Module):
    def __init__(self, base_transformer_model):
        super(Wrapper_Model,self).__init__()
        self.upstream = Upstream_Model(base_transformer_model)
        self.downstream = Downstream_Model()
        
    def forward(self, inputs):
        spec_stacked, pos_enc, attn_mask = inputs
        stacked_encoded_layers = self.upstream(spec_stacked, pos_enc, attn_mask)
        preds = self.downstream(stacked_encoded_layers)
        return preds