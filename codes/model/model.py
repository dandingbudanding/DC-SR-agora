import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)


    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused

class IMDModule_speed_mine(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = sequential(
            nn.conv2d(in_channels, in_channels,kernel_size=(1,3),dilation=(1,2)),
            nn.conv2d(in_channels, in_channels, kernel_size=(3,1), dilation=(2,1)),
        )
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)


    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused
class IMDModule_speed_cca(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed_cca, self).__init__()
        # self.distilled_channels = int(in_channels * distillation_rate)
        # self.remaining_channels = int(in_channels - self.distilled_channels)
        # self.distilled_channels = 3
        # self.remaining_channels = 9
        self.c1 = conv_layer(12, 12, 3)
        self.c2 = conv_layer(9, 12, 3)
        self.c3 = conv_layer(9, 12, 3)
        self.c4 = conv_layer(9, 3, 3)
        self.act = activation('lrelu', neg_slope=0.1)
        self.c5 = conv_layer(12, 12, 1)
        self.cca = CCALayer(12,reduction=4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (3, 9), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (3, 9), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (3, 9), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class IMDModule_speed_cca_ori(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed_cca_ori, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(in_channels,reduction=4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


class IMDModule_speed_cca_ori_prelu(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed_cca_ori_prelu, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act1 = nn.PReLU(num_parameters=in_channels,init=0.2)
        # self.act2 = nn.PReLU(num_parameters=self.remaining_channels, init=0.2)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(in_channels,reduction=4)

    def forward(self, input):
        out_c1 = self.act1(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act1(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act1(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class IMDModule_speed_cca_dyrelu(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed_cca_dyrelu, self).__init__()
        # self.distilled_channels = int(in_channels * distillation_rate)
        # self.remaining_channels = int(in_channels - self.distilled_channels)
        # self.distilled_channels = 3
        # self.remaining_channels = 9
        self.c1 = conv_layer(12, 12, 3)
        self.c2 = conv_layer(9, 12, 3)
        self.c3 = conv_layer(9, 12, 3)
        self.c4 = conv_layer(9, 3, 3)
        self.act = DyReLUA(12)
        self.c5 = conv_layer(12, 12, 1)
        self.cca = CCALayer(12,reduction=4)


    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (3, 9), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (3, 9), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (3, 9), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class IMDModule_speed_cca_dilation(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed_cca_dilation, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3,dilation=2)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3,dilation=2)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3,dilation=2)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3,dilation=2)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4,reduction=4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride,padding=1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)



import torch.nn as nn
import torch

# For any upscale factors
class IMDN_AS(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN_AS, self).__init__()

        self.fea_conv = nn.Sequential(conv_layer(in_nc, nf, kernel_size=3, stride=2),
                                      nn.LeakyReLU(0.05),
                                      conv_layer(nf, nf, kernel_size=3, stride=2))

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

class model_(nn.Module):
    def __init__(self, upscale=2, in_nc=3, n_feats=64, num_modules=6, out_nc=3):
        super(model_, self).__init__()

        self.fea_conv = conv_layer(in_nc, n_feats, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=n_feats)
        self.IMDB2 = IMDModule(in_channels=n_feats)
        self.IMDB3 = IMDModule(in_channels=n_feats)
        self.IMDB4 = IMDModule(in_channels=n_feats)
        self.IMDB5 = IMDModule(in_channels=n_feats)
        self.IMDB6 = IMDModule(in_channels=n_feats)
        self.c = conv_block(n_feats * num_modules, n_feats, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(n_feats, n_feats, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(n_feats, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

# flops: 1.9828800000 GFLOPs, params: 20190.0
class model(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=4, out_nc=3):
        super(model, self).__init__()

        fea_conv = [conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [IMDModule_speed_cca_ori_prelu(in_channels=nf) for _ in range(num_modules)]
        LR_conv = conv_layer(nf, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = nn.Sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                   *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output

# flops: 1.6267399680 GFLOPs, params: 16860.0
class model2(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=4, out_nc=3):
        super(model2, self).__init__()

        fea_conv = [nn.Conv2d(in_nc, nf, kernel_size=3,padding=1)]
        rb_blocks = [IMDModule_speed_cca_ori(in_channels=nf) for _ in range(num_modules)]
        LR_conv = nn.Conv2d(nf, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = nn.Sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output

class model3(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=4, out_nc=3):
        super(model3, self).__init__()

        fea_conv = [conv_layer(in_nc, 12, kernel_size=3)]
        rb_blocks1 = [IMDModule_speed_cca_ori_prelu(in_channels=12) for _ in range(2)]
        LR_conv1 = conv_layer(12, 16, kernel_size=1)
        rb_blocks2 = [IMDModule_speed_cca_ori_prelu(in_channels=16) for _ in range(2)]
        LR_conv2 = conv_layer(16, 12, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks1, LR_conv1, *rb_blocks2, LR_conv2)),
                                *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output

# pass
class model4(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=2, out_nc=3):
        super(model4, self).__init__()

        fea_conv = [conv_layer(in_nc, 12, kernel_size=3)]
        rb_blocks1 = [IMDModule_speed_cca_ori_prelu(in_channels=12) for _ in range(3)]
        LR_conv1 = conv_layer(12, 16, kernel_size=1)
        rb_blocks2 = [IMDModule_speed_cca_ori_prelu(in_channels=16)]
        LR_conv2 = conv_layer(16, 12, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks1, LR_conv1,*rb_blocks2, LR_conv2)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output


# flops: 1.9880650240 GFLOPs, params: 20625.0
class model5(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=5, out_nc=3):
        super(model5, self).__init__()

        fea_conv = [conv_layer(in_nc, 16, kernel_size=3)]
        rb_blocks1 = [IMDModule_speed_cca_ori_prelu(in_channels=16) for _ in range(2)]
        LR_conv1 = conv_layer(16, 12, kernel_size=1)
        rb_blocks2 = [IMDModule_speed_cca_ori_prelu(in_channels=12) for _ in range(2)]
        LR_conv2 = conv_layer(12, 16, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(16, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks1, LR_conv1, *rb_blocks2, LR_conv2)),
                                *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output

class model6(nn.Module):
    def __init__(self, upscale=4, in_nc=3, nf=16, num_modules=6, out_nc=3):
        super(model6, self).__init__()

        fea_conv = [conv_layer(in_nc, 16, kernel_size=3)]
        rb_blocks1 = [IMDModule_speed_cca_ori_prelu(in_channels=16) for _ in range(1)]
        LR_conv1 = conv_layer(16, 12, kernel_size=1)
        rb_blocks2 = [IMDModule_speed_cca_ori_prelu(in_channels=12) for _ in range(3)]
        LR_conv2 = conv_layer(12, 16, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(16, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks1, LR_conv1, *rb_blocks2, LR_conv2)),
                                *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output

class model7(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=4, out_nc=3):
        super(model7, self).__init__()

        fea_conv = [conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [IMDModule_speed(in_channels=nf) for _ in range(num_modules)]
        LR_conv = conv_layer(nf, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output
# baseline: psnr: 30.1618, ssim: 0.8537
# 1.2*2+0.1*4+1.2
