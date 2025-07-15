import os, sys
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录的路径
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到sys.path中
sys.path.append(parent_dir)
import torch.nn as nn
import torch
import pdb
from modules.front_resnet import ResNet34, ResNet34SE, ResNet18, ResNet100, block2module
from modules.pooling import StatsPool, AttentiveStatisticsPooling
# xingming spoof_SE model: LCNN and resnet18
import modules.back_classifier as classifiers
from modules.LightCNN import LightCNN_v1, LightCNN_AP, LightCNN_SE, LightCNN_AP_partition
import modules.pooling as pooling_func
import math
import numpy as np
import random

# class ResNet34StatsPool(nn.Module):
#     def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
#         super(ResNet34StatsPool, self).__init__()
#         self.front = ResNet34(in_planes, **kwargs)
#         self.pool = StatsPool()
#         self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
#         self.drop = nn.Dropout(dropout) if dropout else None
        
#     def forward(self, x, frame_output=False):
#         if frame_output:
#             fo = self.front(x.unsqueeze(dim=1))
#             x = self.pool(fo)
#             x = self.bottleneck(x)
#             if self.drop:
#                 x = self.drop(x)
#             return x, fo.mean(axis=2)
#         else:
#             x = self.front(x.unsqueeze(dim=1))
#             x = self.pool(x)
#             x = self.bottleneck(x)
#             if self.drop:
#                 x = self.drop(x)
#             return x
            


class ResNet34SEStatsPool(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs):
        super(ResNet34SEStatsPool, self).__init__()
        self.front = ResNet34SE(cfg['in_planes'], **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(cfg['in_planes']*8*2, cfg['embedding_size'])
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, xlen=None):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x

class LightCNN_lstm(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs) -> None:
        super(LightCNN_lstm, self).__init__()
        self.front = LightCNN_AP(1,feats_dim=cfg['feats_size'], embd_dim=cfg['embedding_size'])
    
    def forward(self, x, xlen=None):
        x = self.front(x.unsqueeze(dim=1))
        return x

class ResNet18_ASP(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs):
        super(ResNet18_ASP, self).__init__()
        self.front = ResNet18(cfg['in_planes'], **kwargs)
        outmap_size = math.ceil(cfg['feats_size'] / 8 )
        self.pool = getattr(pooling_func, 'ASP_pooling')(cfg['in_planes']*8*outmap_size,cfg['embedding_size'])
        # self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, xlen=None):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        # x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x

class ResNet100_based(nn.Module):
    def __init__(self, cfg, dropout=0.2, **kwargs):
        super(ResNet100_based, self).__init__()
        print('ResNet100 based model with %s and %s ' %(cfg['block_type'], cfg['pooling_layer']))
        self.front = ResNet100(cfg['in_planes'], cfg['block_type'])
        block_expansion = block2module[cfg['block_type']].expansion
        self.pooling = getattr(pooling_func, cfg['pooling_layer'])(cfg['in_planes']*block_expansion,cfg['acoustic_dim'])
        self.bottleneck = nn.Linear(self.pooling.out_dim, cfg['embedding_size'])
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, xlen=None):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        
        return x


from modules.campplus import FCM, TSTP, TDNNLayer, BasicResBlock, CAMDenseTDNNBlock, TransitLayer, get_nonlinear, DenseLayer

from collections import OrderedDict

class CAMPPlus(nn.Module):

    def __init__(self,
                 cfg,
                 feat_dim=80,
                 embed_dim=512,
                 pooling_func='TSTP',
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 **kwargs):
        super(CAMPPlus, self).__init__()

        self.head = FCM(block=BasicResBlock,
                        num_blocks=[2, 2],
                        feat_dim=cfg['acoustic_dim'],)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3),
                                           (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.xvector.add_module('out_nonlinear',
                                get_nonlinear(config_str, channels))

        self.pool = TSTP(in_dim=channels)
        self.pool_out_dim = self.pool.get_out_dim()
        self.xvector.add_module('stats', self.pool)
        self.xvector.add_module(
            'dense',
            DenseLayer(self.pool_out_dim, cfg['embedding_size'], config_str='batchnorm_'))

        # self.classifier = nn.Sequential(nn.BatchNorm1d(cfg['embedding_size']),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(cfg['embedding_size'], 2))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, xlen=None):
        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        # x = self.classifier(x)
        return x

####################EOW-softmax part####################

class ResNet18_ASP_eow(nn.Module):

    def __init__(self, cfg, dropout=0.5, **kwargs):

        super().__init__()
        self.front = ResNet18(cfg['in_planes'], **kwargs)
        outmap_size = math.ceil(cfg['feats_size'] / 8 )
        self.pool = getattr(pooling_func, 'ASP_pooling')(cfg['in_planes']*8*outmap_size,cfg['embedding_size'])
        # self.bottleneck = nn.Linear(cfg['in_planes']*8*2, cfg['embedding_size'])
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, in_embd=False, xlen=None, **kwargs):
        if in_embd == False: # input shpae torch.Size([4, 60, t])
            x = self.front.relu(self.front.bn1(self.front.conv1(x.unsqueeze(dim=1)))) # after conv1 bn1 relu torch.Size([4, 16, 60, t])
            x = self.front.layer1(x) # after layer1 torch.Size([4, 16, 60, t])
            x = self.front.layer2(x) 
            # print(x.shape)
            # after layer2: 2s=torch.Size([4, 32, 30, 126]) 5s=torch.Size([4, 32, 30, 313]) 7s torch.Size([4, 32, 30, 438])
            x = self.front.layer3(x) # after layer3 torch.Size([4, 64, 15, t])
            x = self.front.layer4(x) # after layer4 torch.Size([4, 128, 8, t])
        else:
            # x = self.front.relu(self.bn1(self.conv1(x)))
            # x = self.front.layer1(x)
            # x = self.front.layer2(x)
            x = self.front.layer3(x)
            x = self.front.layer4(x)
        x = self.pool(x) # after pool torch.Size([4, 128])
        # x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x


class LightCNN_eow(nn.Module): # with out lstm layer # edit by yikang for EOW-softmax
    def __init__(self, cfg, dropout=0, **kwargs) -> None:
        super().__init__()
        
        self.front = LightCNN_AP_partition(1,feats_dim=cfg['feats_size'], embd_dim=cfg['embedding_size'], dropout=dropout) # if feats_size = 128, embd_dim = 128
    
    def forward(self, x, in_embd=False, xlen=None, **kwargs):
        if in_embd == False: 
            x = self.front.m_transform_1(x.unsqueeze(dim=1)) 
            # output m_transform_1:  2s=torch.Size([8, 32, 16, 31]) 5s=torch.Size([bs, 32, 16, 78]) 7s=torch.Size([bs, 32, 10, 109])
            # print(x.shape)
            x = self.front.m_transform_2(x) # output m_transform_2:  torch.Size([bs, 32, 8, t]) 
            x = x.permute(0,3,2,1).contiguous() # x after permute:  torch.Size([bs, t, 8, 32])
            bs, frame_len = x.shape[0], x.shape[1]
            lcnn_out = x.view(bs, frame_len, -1) # lcnn output:  torch.Size([8, t, 256])
            # pool = self.front.lstm_pool(lcnn_out) # lstm pooling:  torch.Size([8, t, 256])
        else:
            # x = self.front.m_transform_1(x.unsqueeze(dim=1)) 
            # # output m_transform_1:  2s=torch.Size([8, 32, 16, 31]) 5s=torch.Size([bs, 32, 16, 78]) output m_transform_1:  torch.Size([bs, 32, 16, t])
            x = self.front.m_transform_2(x)
            x = x.permute(0,3,2,1).contiguous()
            bs, frame_len = x.shape[0], x.shape[1]
            lcnn_out = x.view(bs, frame_len, -1)
            # pool = self.front.lstm_pool(lcnn_out)

        out = self.front.m_output(lcnn_out.mean(dim=1)) # output embedding shape:  torch.Size([8, 128])
        return out

# wide resnet For Eow Softmax
from modules.wideresnet import Wide_ResNet
class WideResnet_eow(nn.Module):
    def __init__(self, cfg, depth=22, width=2, norm=None, dropout=0.2, **kwargs):
        super().__init__()
        self.f = Wide_ResNet(depth, widen_factor=width, norm=norm, dropout_rate=dropout, input_channels=1, feats_size=cfg['feats_size'], embedding_size=cfg['embedding_size'])
        
    def forward(self, x, in_embd=False, xlen=None, **kwargs):
        if in_embd == False: # input shape [bs, 60, t]
            out = self.f.conv1(x.unsqueeze(dim=1)) # [bs, 16, 60, t]
            out = self.f.layer1(out) # [bs, 32, 60, t]
            out = out + 0.01 * torch.randn_like(out) # 加噪？高斯噪声 [bs, 32, 60, t]
            out = self.f.layer2(out) # [bs, 64, 30, t]
            # print(out.shape) # [2, 64, 40, 110]
            out = self.f.layer3(out) # [bs, 128, 15, t]
            out = self.f.lrelu(self.f.bn1(out)) # [bs, 128, 15, t]
        else:
            # print(x.shape)
            # out = self.f.conv1(x.unsqueeze(dim=1)) # [bs, 16, 60, t]
            # out = self.f.layer1(out) # [bs, 32, 60, t]
            # out = out + 0.01 * torch.randn_like(out) # 加噪？高斯噪声 [bs, 32, 60, t]
            # out = self.f.layer2(out) 
            # layer2 5s=torch.Size([2, 64, 30, 79]) 
            out = self.f.layer3(x) # [bs, 128, 15, t]
            out = self.f.lrelu(self.f.bn1(out)) # [bs, 128, 15, t]
        
        out = out.permute(0,3,2,1).contiguous() # after permute:torch.Size([bs, t, 15, 128])
        bs, frame_len = out.shape[0], out.shape[1]
        out = out.view(bs, frame_len, -1) #  output:  torch.Size([8, t, 15*128])
        out = self.f.linear2(out.mean(dim=1)) # output embedding shape:  torch.Size([8, 128])
        # 1/0
        return out


# Res18_RFM
class ResNet18_class(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs):
        super().__init__()
        self.front = ResNet18(cfg['in_planes'], **kwargs)
        outmap_size = math.ceil(cfg['feats_size'] / 8 )
        self.pool = getattr(pooling_func, 'ASP_pooling')(cfg['in_planes']*8*outmap_size,cfg['embedding_size'])
        # self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.classifier = getattr(classifiers, 'Linear')(cfg['embedding_size'], 2)
    def forward(self, x, xlen=None):
        x = self.front(x.unsqueeze(dim=1))
        # print(x.shape)
        # 1/0
        x = self.pool(x)
        # print(x.shape)
        # 1/0
        # x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        x = self.classifier(x)
        return x

class Res18_RFM(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs):
        super().__init__()
        # self.spec = torchaudio.transforms.MelSpectrogram(n_mels=80, win_length=int(0.025*16000), hop_length=int(0.01*16000)).cuda()
        # self.EPSILON = torch.tensor(torch.finfo(torch.float).eps, dtype=torch.get_default_dtype()).cuda()
        self.model = ResNet18_class(cfg)

    def cal_fam(self, inputs):
        self.model.zero_grad()
        inputs = inputs.detach().clone()
        inputs.requires_grad_()
        # print(inputs.shape)
        # 1/0
        output = self.model(inputs) #(B,2)
        # print(output.shape)
        # 1/0
        target = output[:, 1]-output[:, 0] #(B,)
        # print(target.shape)
        # 1/0
        target.backward(torch.ones(target.shape).cuda())
        fam = torch.abs(inputs.grad)
        # print(fam.shape) # (4, 80, 701) (B,F,T)
        # 1/0
        c_fam = torch.max(fam, dim=2)[0]
        # print(c_fam.shape) # (4, 80) (B,F)
        # print(c_fam)
        t_fam = torch.max(fam, dim=1)[0]
        # print(t_fam.shape) # (4, 701) (B,T)
        return c_fam, t_fam


    def forward(self, x, is_train):
        # x = self.spec(x)
        # x = torch.max(self.EPSILON, x).log2()
        # x = x - x.mean(dim=2).unsqueeze(dim=2)
        # print(x.shape) # (4, 80, 701) (B,F,T)
        # 1/0
        if is_train:
            self.model.eval()
          
        
            c_mask, t_mask = self.cal_fam(x) # (B,F), (B,T)
            specgmask = torch.ones_like(x) # (B,F,T)
            fillmask = torch.zeros_like(x)
            for i in range(x.size(0)): # 每个batch单独跑
                max_c_ind = np.argsort(c_mask[i].cpu().numpy())[::-1] # 从小到大的序号，倒序*（特征维度的梯度敏感度从大到小）
                # print(max_c_ind) #[79 78 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 20 19 18  8  1 2  3  4  5  6  7  9 17 10 11 12 13 14 15 16 38 39 40 69 62 63 64 65 66 67 68 70 60 71 72 73 74 75 76 77 61 59 41 49 42 43 44 45 46 47 48 50 58 51 52 53 54 55 56 57  0]
                # 1/0
                max_t_ind = np.argsort(t_mask[i].cpu().numpy())[::-1] # 从小到大的序号，倒序*（时间维度的梯度敏感度从大到小）
                # print(len(max_c_ind), len(max_t_ind)) # 80 701
                # 1/0 
                pointcnt = 0
                for (point_c, point_t) in zip(max_c_ind, max_t_ind): #取出序号
                    if specgmask[i][point_c][point_t] == 1: # 第i batch 的 第point_c个特征的第point_t个时间点
                        specgmask[i][point_c, :] = torch.zeros_like(specgmask[i][point_c, :]) # 第i batch 的 第point_c个特征的所有时间点mask, 一整行
                        fillmask[i][point_c, :] = x[i].mean(dim=0) # 填充第i batch 的 第point_c个特征的所有时间点的均值
                        # print(fillmask[i][point_c, :]) # (701,)
                        # 1/0
                        mask_t = random.randint(1, 128)
                        s_t = random.randint(1, mask_t)
                        left = max(point_t-s_t, 0)
                        right = min(point_t+(mask_t-s_t), x.size(2))
                        
                        specgmask[i][:, left:right] = torch.zeros_like(specgmask[i][:, left:right]) # 第i batch 的 第point_t个时间点的所有特征的mask 并增加左右偏置共128列
                        fillmask[i][:, left:right] = x[i].mean(dim=1).unsqueeze(1).expand(-1, right-left)
                        # print(fillmask[i][:, left:right].shape) # (80, <=128)
                        # 1/0
                        pointcnt += 1
                    if pointcnt >= 3:
                        break
            x = specgmask * x + (1-specgmask) * fillmask
            # print(x.shape)
            # 1/0
            self.model.train()
        
        x = self.model(x)
        return x 


class Res18_RFM_narrow(nn.Module):
    def __init__(self, cfg, dropout=0.5, **kwargs):
        super().__init__()
        # self.spec = torchaudio.transforms.MelSpectrogram(n_mels=80, win_length=int(0.025*16000), hop_length=int(0.01*16000)).cuda()
        # self.EPSILON = torch.tensor(torch.finfo(torch.float).eps, dtype=torch.get_default_dtype()).cuda()
        self.model = ResNet18_class(cfg)

    def cal_fam(self, inputs):
        self.model.zero_grad()
        inputs = inputs.detach().clone()
        inputs.requires_grad_()
        # print(inputs.shape)
        # 1/0
        output = self.model(inputs) #(B,2)
        # print(output.shape)
        # 1/0
        target = output[:, 1]-output[:, 0] #(B,)
        # print(target.shape)
        # 1/0
        target.backward(torch.ones(target.shape).cuda())
        fam = torch.abs(inputs.grad)
        # print(fam.shape) # (4, 80, 701) (B,F,T)
        # 1/0
        c_fam = torch.max(fam, dim=2)[0]
        # print(c_fam.shape) # (4, 80) (B,F)
        # print(c_fam)
        t_fam = torch.max(fam, dim=1)[0]
        # print(t_fam.shape) # (4, 701) (B,T)
        return c_fam, t_fam


    def forward(self, x, is_train):
        # x = self.spec(x)
        # x = torch.max(self.EPSILON, x).log2()
        # x = x - x.mean(dim=2).unsqueeze(dim=2)
        # print(x.shape) # (4, 80, 701) (B,F,T)
        # 1/0
        if is_train:
            self.model.eval()

            c_mask, t_mask = self.cal_fam(x) # (B,F), (B,T)
            specgmask = torch.ones_like(x) # (B,F,T)
            fillmask = torch.zeros_like(x)
            for i in range(x.size(0)): # 每个batch单独跑
                max_c_ind = np.argsort(c_mask[i].cpu().numpy())[::-1] # 从小到大的序号，倒序*（特征维度的梯度敏感度从大到小）
                # print(max_c_ind) #[79 78 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 20 19 18  8  1 2  3  4  5  6  7  9 17 10 11 12 13 14 15 16 38 39 40 69 62 63 64 65 66 67 68 70 60 71 72 73 74 75 76 77 61 59 41 49 42 43 44 45 46 47 48 50 58 51 52 53 54 55 56 57  0]
                # 1/0
                max_t_ind = np.argsort(t_mask[i].cpu().numpy())[::-1] # 从小到大的序号，倒序*（时间维度的梯度敏感度从大到小）
                # print(len(max_c_ind), len(max_t_ind)) # 80 701
                # 1/0 
                pointcnt = 0
                for (point_c, point_t) in zip(max_c_ind, max_t_ind): #取出序号
                    if specgmask[i][point_c][point_t] == 1: # 第i batch 的 第point_c个特征的第point_t个时间点
                        specgmask[i][point_c, :] = torch.zeros_like(specgmask[i][point_c, :]) # 第i batch 的 第point_c个特征的所有时间点mask, 一整行
                        fillmask[i][point_c, :] = x[i].mean(dim=0) # 填充第i batch 的 第point_c个特征的所有时间点的均值
                        # print(fillmask[i][point_c, :]) # (701,)
                        # 1/0
                        mask_t = random.randint(1, 62)
                        s_t = random.randint(1, mask_t)
                        left = max(point_t-s_t, 0)
                        right = min(point_t+(mask_t-s_t), x.size(2))
                        
                        specgmask[i][:, left:right] = torch.zeros_like(specgmask[i][:, left:right]) # 第i batch 的 第point_t个时间点的所有特征的mask 并增加左右偏置共128列
                        fillmask[i][:, left:right] = x[i].mean(dim=1).unsqueeze(1).expand(-1, right-left)
                        # print(fillmask[i][:, left:right].shape) # (80, <=128)
                        # 1/0
                        pointcnt += 1
                    if pointcnt >= 3:
                        break
            x = specgmask * x + (1-specgmask) * fillmask
            # print(x.shape)
            # 1/0
            self.model.train()
        
        x = self.model(x)
        return x 


if __name__ == '__main__':
    # x = torch.zeros(2, 80, 14001) #7s sample fft1024 win_length64 hop_length8
    # x = torch.zeros(2, 80, 6401) #4s sample fft512 win_length25 hop_length10
    x = torch.zeros(2, 80, 876).cuda() #7s sample fft1024 win_length400 hop_length128
    from omegaconf import OmegaConf
    # cfg = OmegaConf.load('/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/configs/campplus.yaml')
    # model = CAMPPlus(cfg=cfg['encoder'], dropout=0.4)
    cfg = OmegaConf.load('/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/configs/main_80.yaml')
    # model = ResNet18_ASP(cfg['encoder'], dropout=0.4) 
    # model = LightCNN_eow(cfg['encoder'])
    # model = ResNet18_ASP_eow(cfg['encoder'])
    # model = WideResnet_eow(cfg['encoder'], depth=22, width=2, norm=None, dropout=0.2)
    model = Res18_RFM( cfg['encoder']).cuda()
    model.eval()

    # out = model(x)
    out = model(x, is_train=True)
    print(out.shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
    