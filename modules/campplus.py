# Copyright (c) 2023 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
This implementation is adapted from github repo:
https://github.com/alibaba-damo-academy/3D-Speaker

Some modifications:
1. Reuse the pooling layers in wespeaker
2. Remove the memory_efficient mechanism to meet the torch.jit.script
   export requirements

Reference:
[1] Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng and Qian Chen.
    "CAM++: A Fast and Efficient Network for Speaker Verification
    Using Context-Aware Masking". arXiv preprint arXiv:2303.00332
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torchaudio

class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


class TDNNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, \
                    but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):

    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len: int = 100, stype: str = 'avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x,
                               kernel_size=seg_len,
                               stride=seg_len,
                               ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x,
                               kernel_size=seg_len,
                               stride=seg_len,
                               ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(shape[0], shape[1], shape[2],
                                       seg_len).reshape(
                                           shape[0], shape[1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, \
                but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):

    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels +
                                      i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


'''Note: The stride used here is different from that in Resnet
'''


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):

    def __init__(self, block, num_blocks, m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=2)
        self.layer2 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[1],
                                       stride=2)

        self.conv2 = nn.Conv2d(m_channels,
                               m_channels,
                               kernel_size=3,
                               stride=(2, 1),
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):

    def __init__(self,
                 feat_dim=80,
                 embed_dim=512,
                 pooling_func='TSTP',
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu'):
        super(CAMPPlus, self).__init__()

        self.head = FCM(block=BasicResBlock,
                        num_blocks=[2, 2],
                        feat_dim=feat_dim)
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
            DenseLayer(self.pool_out_dim, embed_dim, config_str='batchnorm_'))

        self.classifier = nn.Sequential(nn.BatchNorm1d(embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(embed_dim, 2))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        # print(x.shape)
        # 1/0
        x = self.classifier(x)
        return x


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(n_mels=80, win_length=int(0.025*16000), hop_length=int(0.01*16000)).cuda()
        self.EPSILON = torch.tensor(torch.finfo(torch.float).eps, dtype=torch.get_default_dtype()).cuda()
        self.model = CAMPPlus().cuda()

    def cal_fam(self, inputs):
        self.model.zero_grad()
        inputs = inputs.detach().clone()
        inputs.requires_grad_()
        # print(inputs.shape)
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
        x = self.spec(x)
        x = torch.max(self.EPSILON, x).log2()
        x = x - x.mean(dim=2).unsqueeze(dim=2)
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
            self.model.train()
        
        x = self.model(x)
        return x 

if __name__ == '__main__':
    # x = torch.zeros(4, 200, 80)
    # model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func='TSTP')
    # model.eval()
    # out = model(x)
    # print(out.shape)

    x = torch.zeros(4, 112000).cuda()
    model = MainModel().cuda()
    model.eval()
    out = model(x, True)
    print(out.shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))