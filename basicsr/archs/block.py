import torch.nn as nn

import torch
import torch.nn.functional as F

# from model.cbam import CBAM


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
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
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

class ESA(nn.Module):
    def __init__(self, n_feats, conv, act_type='relu'):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        if act_type == 'silu':
            self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x * m



class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, act_type='lrelu', use_cbam=False, use_a2n=False, spa_prune=False,
                gct=False, simAM=False):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation(act_type, neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d, act_type)

        self.gct = gct
        if gct:
            self.gct = GCT(num_channels=self.dc*4)
        if use_cbam:
            self.esa = CBAM(in_channels)
        if simAM:
            self.esa = SimAM()
        self.use_a2n = use_a2n
        if self.use_a2n:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.ADM = nn.Sequential(
                nn.Linear(in_channels, in_channels // 10, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // 10, 4, bias=False),
            )
        self.spa_prune = spa_prune
        if spa_prune:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.router = nn.Sequential(
                nn.Linear(in_channels, 3, bias=False)
            )

    def forward(self, input):
        x = input
        a,b,c,d = input.shape
        self.sparsity = 1.0
        if self.spa_prune:
            p = self.avg_pool(input).view(a, b)
            p = self.router(p)
            p = F.softmax(p/3, dim=1)
            idx = torch.mode(torch.argmax(p, dim=1)).values

            prune_scale = [1, 2, 4]
            input = nn.AvgPool2d(prune_scale[idx], stride=prune_scale[idx])(input)
            self.sparsity = prune_scale[idx] * prune_scale[idx]

        with torch.autograd.profiler.record_function("c1_d"):
            distilled_c1 = self.act(self.c1_d(input))
        with torch.autograd.profiler.record_function("c1_r"):
            r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        with torch.autograd.profiler.record_function("c2_d"):
            distilled_c2 = self.act(self.c2_d(r_c1))
        with torch.autograd.profiler.record_function("c2_r"):
            r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        with torch.autograd.profiler.record_function("c3_d"):
            distilled_c3 = self.act(self.c3_d(r_c2))
        with torch.autograd.profiler.record_function("c3_r"):
            r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        with torch.autograd.profiler.record_function("c4"):
            r_c4 = self.act(self.c4(r_c3))

        if self.use_a2n:
            with torch.autograd.profiler.record_function("rfdb_a2n"):
                y = self.avg_pool(input).view(a,b)
                y = self.ADM(y)
                ax = F.softmax(y/4, dim = 1)
                distilled_c1 = distilled_c1 * ax[:,0].view(a,1,1,1)
                distilled_c2 = distilled_c2 * ax[:,1].view(a,1,1,1)
                distilled_c3 = distilled_c3 * ax[:,2].view(a,1,1,1)
                r_c4 = r_c4 * ax[:,3].view(a,1,1,1)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        if self.gct:
            with torch.autograd.profiler.record_function("rfdb_gct"):
                out_fused = self.c5(self.gct(out))
        else:
            with torch.autograd.profiler.record_function("esa"):
                out_fused = self.esa(self.c5(out))

        if self.spa_prune:
            out_fused = F.interpolate(out_fused, scale_factor=prune_scale[idx], mode='nearest')
            out_fused *= p[:, idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            if not self.training:
                outfused = out_fused.resize_(a, b, c, d) + x
            else:
                out_fused = out_fused + x
        return out_fused
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

