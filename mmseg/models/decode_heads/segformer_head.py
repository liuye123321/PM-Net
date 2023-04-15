
import math
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.core import save
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class DGNLTwo(nn.Module):
    def __init__(self, in_channels):
        super(DGNLTwo, self).__init__()
        self.dim = int(in_channels / 2)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.f_phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.f_theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g_1 = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g_3 = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g_2 = nn.Conv2d(in_channels, self.dim, kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)
        #
        self.depth_conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, groups=1, bias=False)
        self.depth_down = nn.Conv2d(1, 1, kernel_size=2, stride=2, groups=1, bias=False)
        self.depth_down.weight.data.fill_(1. / 16)
        self.depth_theta = nn.Conv2d(1, self.dim, kernel_size=1)
        self.depth_phi = nn.Conv2d(1, self.dim, kernel_size=1)
        self.theta = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        original = x
        original_depth = depth_map
        x = self.down(x)
        depth_map = self.depth_down(depth_map)

        g_1 = self.g_1(x).view(n, int(c / 2), -1).transpose(1, 2)
        g_2 = self.g_2(x).view(n, int(c / 2), -1).transpose(1, 2)
        g_3 = self.g_3(x).view(n, int(c / 2), -1).transpose(1, 2)

        theta = self.theta(depth_map).view(n, self.dim, -1).transpose(1, 2)
        phi = self.phi(x).view(n, int(c / 2), -1)
        Ra = F.softmax(torch.bmm(theta, phi), 2)


        d_theta = self.depth_theta(depth_map).view(n, self.dim, -1).transpose(1, 2)
        d_phi = self.depth_phi(depth_map).view(n, self.dim, -1)
        Rb = F.softmax(torch.bmm(d_theta, d_phi), 2)

        f_theta = self.f_theta(x).view(n, self.dim, -1).transpose(1, 2)
        f_phi = self.f_phi(x).view(n, self.dim, -1)
        Rc = F.softmax(torch.bmm(f_theta, f_phi), 2)
        # [n, c / 2, h / 4, w / 4]
        feature_depth = torch.bmm(Ra, g_1).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 2), int(w / 2))
        depth = torch.bmm(Rb, g_2).transpose(1, 2).contiguous().view(n, self.dim, int(h / 2), int(w / 2))
        feature = torch.bmm(Rc, g_3).transpose(1, 2).contiguous().view(n, self.dim, int(h / 2), int(w / 2))
        return original + F.upsample(self.z(feature_depth + depth + feature), size=original.size()[2:], mode='bilinear', align_corners=True)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.n_blocks = 16
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        RCABLayer_c4 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(self.n_blocks)]
        RCABLayer_c3 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(self.n_blocks)]
        RCABLayer_c2 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(self.n_blocks)]
        RCABLayer_c1 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(self.n_blocks)]
        self.RCABLayer_c4 = nn.Sequential(*RCABLayer_c4)
        self.RCABLayer_c3 = nn.Sequential(*RCABLayer_c3)
        self.RCABLayer_c2 = nn.Sequential(*RCABLayer_c2)
        self.RCABLayer_c1 = nn.Sequential(*RCABLayer_c1)

        # RCABLayer = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(self.n_blocks)]
        # self.RCABLayer = nn.Sequential(*RCABLayer)

        self.upsample_c4 = Upsampler(conv=default_conv, scale=8, n_feats=embedding_dim, act='relu')
        self.upsample_c3 = Upsampler(conv=default_conv, scale=4, n_feats=embedding_dim, act='relu')
        self.upsample_c2 = Upsampler(conv=default_conv, scale=2, n_feats=embedding_dim, act='relu')


        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        # upsample_c_x2 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(4)]
        # upsample_c_x2.append(Upsampler(conv=default_conv, scale=2, n_feats=embedding_dim, act='relu'))
        # self.upsample_c_x2 = nn.Sequential(*upsample_c_x2)
        upsample_c_x4 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(4)]
        upsample_c_x4.append(Upsampler(conv=default_conv, scale=4, n_feats=embedding_dim, act='relu'))
        self.upsample_c_x4 = nn.Sequential(*upsample_c_x4)
        upsample_c_x8 = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(4)]
        upsample_c_x8.append(Upsampler(conv=default_conv, scale=2, n_feats=embedding_dim, act='relu'))
        self.upsample_c_x8 = nn.Sequential(*upsample_c_x8)

        # upsample_c = [RCAB(conv=default_conv, n_feat=embedding_dim, kernel_size=3) for _ in range(4)]
        # self.upsample_c = nn.Sequential(*upsample_c)
        # upsample_c_x4 = Upsampler(conv=default_conv, scale=4, n_feats=embedding_dim, act='relu')
        # self.upsample_c_x4 = nn.Sequential(*upsample_c_x4)
        # upsample_c_x8 = Upsampler(conv=default_conv, scale=2, n_feats=embedding_dim, act='relu')
        # self.upsample_c_x8 = nn.Sequential(*upsample_c_x8)

        self.tail = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1
        )
    def forward(self, inputs, haze_input):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

            ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.RCABLayer_c4(_c4)
        _c4 = self.upsample_c4(_c4)
        # _c4 = resize(
        #         input=_c4,
        #         size=c1.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.RCABLayer_c3(_c3)
        _c3 = self.upsample_c3(_c3)
        # _c3 = resize(
        #         input=_c3,
        #         size=c1.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.RCABLayer_c2(_c2)
        _c2 = self.upsample_c2(_c2)
        # _c2 = resize(
        #         input=_c2,
        #         size=c1.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = self.RCABLayer_c1(_c1)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        haze = self.dropout(_c)
        haze = self.upsample_c_x8(self.upsample_c_x4(haze))
        # haze = self.upsample_c_x4(haze)

        dehaze = self.tail(haze) + haze_input
        out = {'haze': dehaze}
        return out

    # def forward(self, inputs, haze_input):
    #     x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
    #     c1, c2, c3, c4 = x
    #
    #         ############## MLP decoder on C1-C4 ###########
    #     n, _, h, w = c4.shape
    #     _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
    #     _c4 = self.RCABLayer(_c4)
    #     _c4 = self.upsample_c4(_c4)
    #
    #     _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
    #     _c3 = self.RCABLayer(_c3)
    #     _c3 = self.upsample_c3(_c3)
    #
    #     _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
    #     _c2 = self.RCABLayer(_c2)
    #     _c2 = self.upsample_c2(_c2)
    #
    #     _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
    #     _c1 = self.RCABLayer(_c1)
    #     _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
    #
    #     haze = self.dropout(_c)
    #     haze = self.upsample_c_x8(self.upsample_c(self.upsample_c_x4(self.upsample_c(haze))))
    #     dehaze = self.tail(haze) + haze_input
    #     out = {'haze': dehaze}
    #
    #     return out