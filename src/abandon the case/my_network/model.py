import torch.nn as nn
import torch

import os

from .backbone import EfficientNet as EffNet
from .utils import MemoryEfficientSwish, Swish
from .utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from dcn_v2 import DCN
from dcn_v2 import DCNv2Pooling
import torchvision

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x




class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """
    @staticmethod
    def get_img_size(compound_coef):
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        return input_sizes[compound_coef]
    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        model = EffNet.from_pretrained(f'efficientnet-b{self.backbone_compound_coef[compound_coef]}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps

class MyBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(MyBlock, self).__init__()
        class CBR(torch.nn.Module):
            def __init__(self,inchannel,outchannel):
                super(CBR, self).__init__()
                self.cbr = torch.nn.Sequential(
                    DCN(inchannel, outchannel, 3, 1, 1),
                    torch.nn.BatchNorm2d(outchannel),
                    torch.nn.ReLU(True),
                )
            def forward(self,x):
                return self.cbr(x)
        self.cbr1 = CBR(inchannel,outchannel)
        self.resconn = torch.nn.Conv2d(inchannel,outchannel,1,1)
        self.cbr2 = CBR(outchannel,outchannel)
    def forward(self,x):
        tmp = self.cbr1(x)
        return  self.cbr2(tmp) + self.resconn(x)
class Net(nn.Module):
    def __init__(self,compound_coef,class_num):
        super(Net, self).__init__()
        self.backbone = EfficientNet(compound_coef)
        self.blocks = torch.nn.Sequential()
        self.spatial_scale = 0.25
        out_channels = 32
        for i in range(2):
            self.blocks.add_module('Myblock-{}'.format(i),MyBlock(out_channels,out_channels*2))
            out_channels *= 2
        self.classifier = torch.nn.Linear(out_channels,class_num)
    def load_weights(self,path):
        self.backbone.load_state_dict(torch.load(path),strict=False)
    def forward(self, x):
        img = x['img']
        batch_size = img.size()[0]
        tlbrs = x['tlbrs']
        ids = x['ids']
        max_objs = tlbrs.size()[1]
        tlbrs = torch.cat((torch.ones(batch_size,max_objs,1).to(tlbrs.device),tlbrs),dim=-1).float()

        for i in range(batch_size):
            tlbrs[i,:,0] = i
        tlbrs = tlbrs.reshape(batch_size*max_objs,5)
        ids = ids.reshape(-1)
        tlbrs = tlbrs[ids != -1]
        ids = ids[ids!=-1]
        x['tlbrs'] = tlbrs
        x['ids'] = ids
        featmap = self.backbone(img)[-1]
        featmap = self.blocks(featmap)
        features = torchvision.ops.roi_align(featmap,tlbrs,1,self.spatial_scale,4)
        features = features.reshape(len(ids),-1)

        # features = torch.nn.functional.normalize(features,2,-1)

        return  self.classifier(features),features








if __name__ == '__main__':
    compound_coef = 8
    model = Net(compound_coef)
    model.cuda()
    model.load_weights('G:\project\MOT\Model\efficientnet-d{}.pth'.format(compound_coef))
    model.eval()
    with torch.no_grad():
        imgsize = EfficientNet.get_img_size(compound_coef)
        # img = torch.rand((2,3,imgsize,imgsize)).cuda()
        model([torch.rand((1,3,imgsize,imgsize)).cuda().requires_grad_(False),None])
        input(imgsize)