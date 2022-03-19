from .. import  transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self.out = nn.Linear(self.d_model, self.trg_vocab)
        # self.selfencoder = transformer.Models.Encoder(self.src_vocab,self.d_model, self.N, self.heads, self.dropout)
        # self.cmpout = nn.Linear(self.d_model,self.trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        def func(src,trg,src_mask,trg_mask):
            e_outputs = self.encoder(src, src_mask)
            d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
            output = self.out(d_output)
            output = F.normalize(output, 2, -1)
            return  output

        out1 = func(src,trg,src_mask,trg_mask)
        out2 = func(trg,src,trg_mask,src_mask)
        # out1 = F.normalize(src,2,-1)
        # out2 = F.normalize(trg,2,-1)
        mat = torch.matmul(out1,out2.transpose(2,1))
        src_mask = src_mask.unsqueeze(-1)
        trg_mask = trg_mask.unsqueeze(1)
        sorce_mat = torch.matmul(src_mask, trg_mask)
        similar = torch.sum(mat*sorce_mat,(-1,-2)) / torch.sum(sorce_mat,(-1,-2))

        return 1 - (similar + 1) / 2

class Myloss(nn.Module):
    def __init__(self,model_config):
        super(Myloss, self).__init__()
        self.dis = model_config['dis']
        self.loss_dict = {'loss':None,'p':None,'n':None,'ap':None}
        self.map_com_itels = model_config['map_com_itels']
        self.ps = []
        self.ns = []
        self.num = 0
    def forward(self,p,n):
        loss = torch.clip(p-n+self.dis,0,1)
        self.loss_dict['loss'] = loss.clone().detach().cpu()
        self.loss_dict['p'] = p.clone().detach().cpu()
        self.loss_dict['n'] = n.clone().detach().cpu()
        self.ps += np.array(self.loss_dict['p']).tolist()
        self.ns += np.array(self.loss_dict['n']).tolist()
        self.num += 1
        if self.num % self.map_com_itels == 0:
            self.loss_dict['ap'] = self.compuate()
        return loss
    def compuate(self):
        def voc_ap(rec, prec, use_07_metric=False):
            """ ap = voc_ap(rec, prec, [use_07_metric])
            Compute VOC AP given precision and recall.
            If use_07_metric is true, uses the
            VOC 07 11 point method (default:False).
            """
            # 针对2007年VOC，使用的11个点计算AP，现在不使用
            if use_07_metric:
                # 11 point metric
                ap = 0.
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    ap = ap + p / 11.
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.], rec, [1.]))  # [0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
                mpre = np.concatenate(([0.], prec, [0.]))  # [0.  1.,     0.6666, 0.4285, 0.3043,  0.]

                # compute the precision envelope
                # 计算出precision的各个断点(折线点)
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # [1.     1.     0.6666 0.4285 0.3043 0.    ]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]  # precision前后两个值不一样的点

                # AP= AP1 + AP2+ AP3+ AP4
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap
        sorted(self.ps)
        sorted(self.ns)
        i,j=0,0
        tp = 0
        fp = 0
        fn = len(self.ps)
        rec = []
        prec = []
        while i < len(self.ps) and j < len(self.ns):
            if self.ps[i] <= self.ns[j]:
                tp += 1
                fn -= 1
                i+=1
            else:
                fp += 1
                j += 1
            prec.append(tp/(tp+fp))
            rec.append(tp/(tp+fn))
        while i < len(self.ps):
            tp += 1
            fn -= 1
            i += 1
            prec.append(tp / (tp + fp))
            rec.append(tp / (tp + fn))
        while j < len(self.ns):
            fp += 1
            j += 1
            prec.append(tp / (tp + fp))
            rec.append(tp / (tp + fn))
        return voc_ap(np.array(rec),np.array(prec))
    def reset_ap(self):
        self.ps = []
        self.ns = []