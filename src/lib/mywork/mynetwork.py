from .. import  transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
class Mynetwork(nn.Module):
    def __init__(self,config):
        super(Mynetwork, self).__init__()
        self.src_vocab = config['src_vocab']
        self.trg_vocab = config['trg_vocab']
        self.d_model = config['d_model']
        self.N = config['N']
        self.heads = config['heads']
        self.dropout = config['dropout']
        self.encoder = transformer.Models.Encoder(self.src_vocab,self.d_model, self.N, self.heads, self.dropout)
        self.decoder = transformer.Models.Decoder(self.trg_vocab, self.d_model, self.N, self.heads, self.dropout)
        self.selfencoder = transformer.Models.Encoder(self.src_vocab,self.d_model, self.N, self.heads, self.dropout)
        self.out = nn.Linear(self.d_model, self.trg_vocab)
        self.cmpout = nn.Linear(self.d_model,self.trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        output = F.normalize(output, 2, -1)

        se_output = self.selfencoder(trg, trg_mask)
        se_output = self.cmpout(se_output)
        se_output = F.normalize(se_output, 2, -1)

        similar = torch.sum(torch.sum(output * se_output,-1) * trg_mask, -1) / torch.sum(trg_mask, -1)
        return 1 - (similar + 1) / 2

class Myloss(nn.Module):
    def __init__(self,model_config):
        super(Myloss, self).__init__()
        self.dis = model_config['dis']
        self.loss_dict = {'loss':None,'p':None,'n':None}
    def forward(self,p,n):
        loss = torch.clip(p-n+self.dis,0,1) + p
        self.loss_dict['loss'] = loss.clone().detach().cpu()
        self.loss_dict['p'] = p.clone().detach().cpu()
        self.loss_dict['n'] = torch.clip(p-n+self.dis,0,1).clone().detach().cpu()
        return loss
