import torch 
class PositionalEncoding(torch.nn.Module): 
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model 

        '''
            pe(pos, 2i) = sin(pos/ 10000^2i/d)
            pe(pos, 2i+1) = cos(pos/ 10000^2i/d)
        '''
        pe = torch.zeros([self.seq_len, self.d_model])

        pos = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(dim=1) # (seq_len, 1)
        idx = torch.arange(self.d_model, dtype=torch.float).unsqueeze(dim=0) # (1, d_model)

        angle_rates = 1/10000**(2*(idx//2)/d_model)
        angle_rads = pos * angle_rates

        pe[:, 0::2] = torch.sin(angle_rads) # even indices
        pe[:, 1::2] = torch.cos(angle_rads) # odd indices

        pe = pe.unsqueeze(0)  # Shape: [1, seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        seq_len = input.size(1) # (batch_size, seq_len, d_model)
        return input + self.pe[:, :seq_len, :] 


