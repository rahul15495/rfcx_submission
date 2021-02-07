import torch
import numpy as np
from functools import lru_cache



class Preprocessor:
    def __init__(self, hidden_size, dr, device):
        self.hidden_size = hidden_size
        self.device = device
        self.dr= dr

    def down_sample_frames(self, spec):
        left_over = spec.shape[1] % self.dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
        return spec_stacked

    @staticmethod
    @lru_cache(maxsize=128)
    def get_sinusoid_table(seq_len,hidden_size):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

    @staticmethod
    def fast_position_encoding(seq_len,hidden_size, batch_size=None):
        ''' position encoding table '''
        table = Preprocessor.get_sinusoid_table(seq_len,hidden_size)

        if batch_size is not None:
            # using expand will not cause extra CPU memory allocation issue
            # however, the expanded tensor after put into GPU still need
            # GPU memory of expanded size, which should be avoided when
            # positional table is large
            # this path is not recommended
            batch_table = table.expand(batch_size, -1, -1)
            return batch_table # (batch_size, seq_len, hidden_size)
        else:
            # this path is most recommended, no extra CPU and GPU memory allocation
            # after getting the (seq_len, hidden_size) tensor, one should first put
            # this tensor into GPU then expand it
            return table  # (seq_len, hidden_size)


    def process_MAM_data(self, spec):
            """Process testing data for the masked acoustic model"""

            # Hack bucket if spec is loaded from the dataloader
            if len(spec.shape) == 4: # Bucketing should cause acoustic feature to have shape 1xBxTxD
                spec = spec.squeeze(0)
            # add arbitary batch axis B if input `spec` has shape of TxD
            elif len(spec.shape) == 2:
                spec = spec.unsqueeze(0)
            # input `spec` should have shape BxTxD
            elif len(spec.shape) != 3:
                raise ValueError('Input argument `spec` has invalid shape: {}'.format(spec.shape))

            # Down sample
            spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, mel_dim * dr)

            # Record length for each uttr
            spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
            spec_len = [int(sl) for sl in spec_len]

            batch_size = spec_stacked.shape[0]
            seq_len = spec_stacked.shape[1]

            pos_enc = Preprocessor.fast_position_encoding(seq_len,self.hidden_size)
            attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

            # zero vectors for padding dimension
            for idx in range(len(spec_stacked)):
                attn_mask[idx][spec_len[idx]:] = 0 

            spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
            pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32).expand(spec_stacked.size(0), *pos_enc.size())
            attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
            return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)