import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs.block as B

def make_model(args, parent=False):
    model = RFDN(nf=args.n_feats, upscale=args.scale[0], from_t=args.from_teacher, act_type=args.act,
                ds_rate=args.ds_rate, use_cbam=args.cbam, use_a2n=args.a2n)
    return model

@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4, from_t=False, act_type='lrelu',
                ds_rate=0.25, use_cbam=False, use_a2n=False, rfdb_a2n=False, spa_prune=False, gct=False,
                simam=False):
        super(RFDN, self).__init__()
        self.from_t = from_t
        self.num_modules = num_modules
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.gct = gct
        if gct:
            self.gct_in = B.GCT(num_channels=nf)

        self.B1 = B.RFDB(in_channels=nf, act_type=act_type, distillation_rate=ds_rate, use_cbam=use_cbam, use_a2n=rfdb_a2n, spa_prune=spa_prune,
                        gct=gct, simAM=simam)
        self.B2 = B.RFDB(in_channels=nf, act_type=act_type, distillation_rate=ds_rate, use_cbam=use_cbam, use_a2n=rfdb_a2n, spa_prune=spa_prune,
                        gct=gct, simAM=simam)
        self.B3 = B.RFDB(in_channels=nf, act_type=act_type, distillation_rate=ds_rate, use_cbam=use_cbam, use_a2n=rfdb_a2n, spa_prune=spa_prune,
                        gct=gct, simAM=simam)
        if num_modules == 4:
            self.B4 = B.RFDB(in_channels=nf, act_type=act_type, distillation_rate=ds_rate, use_cbam=use_cbam, use_a2n=rfdb_a2n, spa_prune=spa_prune,
                        gct=gct, simAM=simam)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)

        self.use_a2n = use_a2n
        if self.use_a2n:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.ADM = nn.Sequential(
                nn.Linear(nf, nf // 10, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nf // 10, num_modules, bias=False),
            )
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        with torch.autograd.profiler.record_function("fea_conv"):
            out_fea = self.fea_conv(input)
        if self.gct:
            with torch.autograd.profiler.record_function("gct_in"):
                out_fea = self.gct_in(out_fea)
        a, b, c, d = out_fea.shape
        with torch.autograd.profiler.record_function("B1"):
            out_B1 = self.B1(out_fea)
        with torch.autograd.profiler.record_function("B2"):
            out_B2 = self.B2(out_B1)
        with torch.autograd.profiler.record_function("B3"):
            out_B3 = self.B3(out_B2)
        sparsity = self.B1.sparsity + self.B2.sparsity + self.B3.sparsity
        if self.num_modules == 4:
            with torch.autograd.profiler.record_function("B4"):
                out_B4 = self.B4(out_B3)
                sparsity += self.B4.sparsity
        sparsity = sparsity / self.num_modules

        if self.use_a2n:
            with torch.autograd.profiler.record_function("A2N"):
                y = self.avg_pool(out_fea).view(a,b)
                y = self.ADM(y)
                ax = F.softmax(y/self.modules, dim = 1)
                out_B1 = out_B1 * ax[:,0].view(a,1,1,1)
                out_B2 = out_B2 * ax[:,1].view(a,1,1,1)
                out_B3 = out_B3 * ax[:,2].view(a,1,1,1)
                if self.num_modules == 4:
                    out_B4 = out_B4 * ax[:,3].view(a,1,1,1)

        if self.num_modules == 4:
            out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        elif self.num_modules == 3:
            out_B = self.c(torch.cat([out_B1, out_B2, out_B3], dim=1))
        with torch.autograd.profiler.record_function("LR_conv"):
            out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        # return output, sparsity
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # if self.from_t:
            #     print(type(param), param.shape)
            #     param = param[1:]
            if name in own_state:
                if self.from_t:
                    if len(own_state[name].shape) == 1:
                        param = param[:own_state[name].shape[0]]
                    else:
                        # print(name, param.shape, own_state[name].shape)
                        param = param[:own_state[name].shape[0], :own_state[name].shape[1], :, :]
                        # print(param.shape)

                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('upsampler') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('upsampler') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
