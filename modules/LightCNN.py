import torch 
import sys 
import torch.nn as nn


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


# For blstm
class BLSTMLayer(nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, \
                                     bidirectional=True)
    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)

class LightCNN_v1(nn.Module):
    def __init__(self,in_channel=1, feats_dim=80, embd_dim=128, dropout=0.5) -> None:
        super(LightCNN_v1, self).__init__()
        self.m_transform = []
        self.m_transform.append(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, [5, 5], 1, padding=[2, 2]),
                MaxFeatureMap2D(),
                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
                nn.BatchNorm2d(48, affine=False),

                nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(48, affine=False),
                nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.MaxPool2d([2, 2], [2, 2]),
                
                nn.Dropout(dropout)
            )
        )
        self.m_transform = nn.Sequential(*self.m_transform)


        self.lstm_pool = nn.Sequential(
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32),
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32)
        )

        # self.m_output = nn.Linear((feats_dim//16) * 32, embd_dim)

    def forward(self, x):
        
        x = self.m_transform(x)
        x = x.permute(0,3,1,2).contiguous()
        btsz ,frame_len = x.shape[0],x.shape[1]
        hidden = x.view(btsz, frame_len, -1)
        x_h = self.lstm_pool(hidden)
        x = (x_h+hidden)
        return x
    
########LCNN from STC team (also ASVspoof2021 Baseline )########
class LightCNN_AP(nn.Module): # frame-level average pooling [asvspoof2021 baseline]
    def __init__(self,in_channel=1, feats_dim=80, embd_dim=128, dropout=0.5) -> None:
        super(LightCNN_AP, self).__init__()
        self.m_transform = []
        self.m_transform.append(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, [5, 5], 1, padding=[2, 2]),
                MaxFeatureMap2D(),
                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
                nn.BatchNorm2d(48, affine=False),

                nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(48, affine=False),
                nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.MaxPool2d([2, 2], [2, 2]),
                
                nn.Dropout(dropout)
            )
        )
        self.m_transform = nn.Sequential(*self.m_transform)

        self.lstm_pool = nn.Sequential(
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32),
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32)            
        )

        self.m_output = nn.Linear((feats_dim//16) * 32, embd_dim)

    def forward(self, x):
        # print(x.shape) # batch, channel, feat, time e.g.: torch.Size([256, 1, 128, 387])
        x = self.m_transform(x) # batch, channel, feature, time e.g.: torch.Size([256, 32, 8, 24])
        # print(x.shape)
        x = x.permute(0,3,1,2).contiguous()
        btsz ,frame_len = x.shape[0],x.shape[1]
        # print(btsz,frame_len)
        hidden = x.view(btsz, frame_len, -1) # batch, time, feature e.g.: torch.Size([256, 24, 256])
        # print(hidden.shape)
        x_h = self.lstm_pool(hidden)
        # print(x_h.shape)
        x = self.m_output((x_h+hidden).mean(dim=1))
        # exit()
        return x


class LightCNN_AP_partition(nn.Module): # edit by yikang for EOW-softmax
    def __init__(self,in_channel=1, feats_dim=80, embd_dim=128, dropout=0.5) -> None:
        super().__init__()
        self.m_transform = []
        self.m_transform.append(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, [5, 5], 1, padding=[2, 2]),
                MaxFeatureMap2D(),
                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
                nn.BatchNorm2d(48, affine=False), # batchNorm_10

                nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(48, affine=False),
                nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]), # conv_20
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False)
            )
        )
        self.m_transform_1 = nn.Sequential(*self.m_transform)
        self.m_transform_2 = nn.Sequential(
            nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(), # MFM_27
            nn.MaxPool2d([2, 2], [2, 2]), # MaxPool_28
            nn.Dropout(dropout)
        )


        self.lstm_pool = nn.Sequential(
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32),
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32)            
        )

        self.m_output = nn.Linear((feats_dim//16) * 32, embd_dim)

    def forward(self, x):
        # print(x.shape) # batch, channel, feat, time e.g.: torch.Size([256, 1, 128, 387])
        x = self.m_transform_1(x) 
        x = self.m_transform_2(x) # batch, channel, feature, time e.g.: torch.Size([256, 32, 8, 24])
        # print(x.shape)
        x = x.permute(0,3,1,2).contiguous()
        btsz ,frame_len = x.shape[0],x.shape[1]
        # print(btsz,frame_len)
        hidden = x.view(btsz, frame_len, -1) # batch, time, feature e.g.: torch.Size([256, 24, 256])
        # print(hidden.shape)
        x_h = self.lstm_pool(hidden)
        # print(x_h.shape)
        x = self.m_output((x_h+hidden).mean(dim=1))
        # exit()
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LightCNN_SE(nn.Module):
    def __init__(self,in_channel=1, feats_dim=80, embd_dim=128, dropout=0.5) -> None:
        super(LightCNN_SE, self).__init__()
        self.m_transform = []
        self.m_transform.append(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, [5, 5], 1, padding=[2, 2]),
                MaxFeatureMap2D(),
                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
                nn.BatchNorm2d(48, affine=False),
                SELayer(48),

                nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(48, affine=False),
                nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                SELayer(32),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                SELayer(32),
                nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.MaxPool2d([2, 2], [2, 2]),
                
                nn.Dropout(dropout)
            )
        )
        self.m_transform = nn.Sequential(*self.m_transform)


        self.lstm_pool = nn.Sequential(
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32),
            BLSTMLayer((feats_dim//16) * 32, (feats_dim//16) * 32)            
        )

        # self.m_output = nn.Linear((feats_dim//16) * 32, embd_dim)

    def forward(self, x):
        
        x = self.m_transform(x)
        x = x.permute(0,3,1,2).contiguous()
        btsz ,frame_len = x.shape[0],x.shape[1]
        hidden = x.view(btsz, frame_len, -1)
        x_h = self.lstm_pool(hidden)
        x = (x_h+hidden)
        return x
    
