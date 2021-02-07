s3prl_path= '/home/jupyter/rfcx/s3prl'
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
            
            sub_sequences= []
            num_subseq= sequence_length//self.maxlen
            start= 0
            
            for i in range(1,num_subseq+1):
                end= self.maxlen*i
                if axes_length==2:
                    sub_sequences.append(inp[:, start:end])
                else:
                    sub_sequences.append(inp[:, start:end, :])
                
                start=end
                
            if end<sequence_length:
                if axes_length==2:
                    sub_sequences.append(inp[:, start:])
                else:
                    sub_sequences.append(inp[:, start:, :])
        
            return sub_sequences
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
    for eg: (1,3,4801, 768)
    '''
    
    def __init__(self, encoder_num_layers=3, frame_length=4801, hidden_layers=768, num_classes=24):
        
        super(Downstream_Model,self).__init__()
        
        self.num_layers  = encoder_num_layers
        self.weight      = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        self.drop1        = nn.Dropout()
        self.pool        = nn.AvgPool2d(kernel_size= (frame_length, 1))
        self.drop2       = nn.Dropout()
        self.hidden1     = nn.Linear(in_features= hidden_layers, out_features =128)
        self.act1        = nn.ReLU()
        self.drop3       = nn.Dropout()
        self.logits      = nn.Linear(in_features= 128, out_features =num_classes)
        self.classifier   = nn.Sigmoid()
        
    def forward(self, x):
        
        #weighted_sum. expected_output (Batch_Size, Time_step , Hidden_Size_Of_Upstream)
        softmax_weight = nn.functional.softmax(self.weight, dim=-1)
        x_weighted_sum = torch.matmul(x,softmax_weight)
        x_weighted_sum = self.drop1(x_weighted_sum)
        
        # frame-wise average pool. expected output (Batch_Size, 1 , Hidden_Size_Of_Upstream) 
        golbal_activations = self.pool(x_weighted_sum)
        golbal_activations = golbal_activations.squeeze(dim=1) #(Batch_Size, Hidden_Size_Of_Upstream) 
        golbal_activations = self.drop2(golbal_activations)
        
        hidden_layer = self.hidden1(golbal_activations)
        hidden_layer = self.act1(hidden_layer)
        hidden_layer = self.drop3(hidden_layer)
        
        logits_layer = self.logits(hidden_layer)
        return self.classifier(logits_layer)
    
class Wrapper_Model(torch.nn.Module):
    def __init__(self, base_transformer_model):
        super(Wrapper_Model,self).__init__()
        self.upstream = Upstream_Model(base_transformer_model)
        self.downstream = Downstream_Model()
        
    def forward(self, spec_stacked, pos_enc, attn_mask):
        stacked_encoded_layers = self.upstream(spec_stacked, pos_enc, attn_mask)
        preds = self.downstream(stacked_encoded_layers)
        return preds