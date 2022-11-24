import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from . import fft
from . import lie_tools
from . import utils
from . import lattice
from . import mrc
from . import symm_groups
from . import models

class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownBlock, self).__init__()

        down_block = []
        down_block.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
        down_block.append(nn.LeakyReLU(0.2))
        #down_block.append(nn.Conv3d(outchannels, outchannels, 3, 1, 1))
        #down_block.append(nn.LeakyReLU(0.2))
        self.down = nn.Sequential(*down_block)
        utils.initseq(self.down)

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()

        up_block = []
        #up_block.append(nn.Conv3d(inchannels, outchannels, 3, 1, 1))
        #up_block.append(nn.LeakyReLU(0.2))
        up_block.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
        up_block.append(nn.LeakyReLU(0.2))

        self.up = nn.Sequential(*up_block)
        utils.initseq(self.up)

    def forward(self, x):
        return self.up(x)

class Unet(nn.Module):
    def __init__(self, D, zdim=256):

        super(Unet, self).__init__()
        self.vol_size = D - 1
        self.crop_vol_size = 160
        #downsample volume
        self.start_size = 64
        self.down_out_dim = (self.start_size)//32
        self.zdim = zdim
        self.transformer = models.SpatialTransformer(self.start_size)

        self.up_outchannels =   {2:128, 4:192, 8:160, 16:128, 32:64, 64:32, 128:16, 256:1}
        self.down_outchannels = {2:256, 4:192, 8:160, 16:128, 32:64, 64:32}
        self.concats = [2, 4, 8]

        downs = []
        resolution = 64
        n_layers = int(np.log2(resolution) - 1)
        inchannels = 1

        for i in range(n_layers):
            resolution //= 2
            outchannels = self.down_outchannels[resolution]
            downs.append(DownBlock(inchannels, outchannels))
            inchannels = outchannels

        self.downs = nn.ModuleList(downs)
        self.downz = nn.Sequential(
                nn.Linear(self.down_outchannels[2] * self.down_out_dim ** 3, 512), nn.LeakyReLU(0.2))

        self.mu = nn.Linear(512, self.zdim)
        #self.logstd = nn.Linear(512, self.zdim)

        utils.initseq(self.downz)

        utils.initmod(self.mu)
        #utils.initmod(self.logstd)

        self.upz = nn.Sequential(nn.Linear(self.zdim, 1024), nn.LeakyReLU(0.2))
        utils.initseq(self.upz)
        ups = []
        inchannels = 128
        resolution = 2
        self.templateres = 256

        for i in range(int(np.log2(self.templateres)) - 1):
            up_block = []
            if resolution in self.concats:
                inchannels += self.down_outchannels[resolution]
            resolution *= 2
            outchannels = self.up_outchannels[resolution]
            ups.append(UpBlock(inchannels, outchannels))
            #if inchannels == outchannels:
            inchannels = outchannels
            #outchannels = max(inchannels // 2, 16)
            #else:
            #inchannels = outchannels
        self.ups = nn.ModuleList(ups)

        #create 2d grid for translation
        x_idx = torch.arange(0, self.start_size, dtype=torch.float32)
        grids = torch.meshgrid(x_idx, x_idx)
        self.register_buffer("grid_2d",  torch.stack((grids[1], grids[0]), dim=-1).unsqueeze(0))

    def translate(self, x, trans):
        B = trans.shape[0]
        trans_dim = trans.shape[1]
        grid_t = self.grid_2d - trans.view((B, 1, 1, trans_dim))
        # put in range -1, 1
        grid_t = 2.*(grid_t/float(self.crop_vol_size - 1.) - 0.5)
        #images (B, 1, H, W)
        translated = F.grid_sample(x, grid_t, align_corners=True)
        return translated

    def forward(self, img, rots, trans, losslist=["kldiv"]):
        B = img.shape[0]
        start = (self.vol_size - self.crop_vol_size)//2
        tail = start + self.crop_vol_size
        x = img[:, :, start:tail, start:tail]
        x = self.translate(x, trans)
        x3d = x.repeat(1, self.crop_vol_size, 1, 1) #(N, D, H, W)

        down_encs = []
        for i in range(B):
            pos = self.transformer.rotate(rots[i].T)
            #downsample x3d
            x3d_down = F.grid_sample(x3d[i:i+1].unsqueeze(1), pos, align_corners=True)
            #down64 = self.downs[0](x3d_down) #(32)
            down32 = self.downs[0](x3d_down)  #64
            down_encs.append(down32)

        down_encs = torch.cat(down_encs, dim=0)
        down16 = self.downs[1](down_encs)  #128
        down8  = self.downs[2](down16) #256
        down4  = self.downs[3](down8) #512
        down2  = self.downs[4](down4) #512
        downz  = self.downz(down2.view(-1, self.down_out_dim ** 3 * self.down_outchannels[2]))
        z      = self.mu(downz)
        #upsample x
        up2   = self.upz(z).view(-1, 128, 2, 2, 2) #512
        #print(up2.shape, down2.shape)
        feat2 = torch.cat([down2, up2], dim=1)
        up4   = self.ups[0](feat2) #512
        #print(up4.shape)
        feat4 = torch.cat([down4, up4], dim=1)
        up8   = self.ups[1](feat4) #256
        feat8 = torch.cat([down8, up8], dim=1)
        up16  = self.ups[2](feat8) #128
        #feat16 = torch.cat([down16, up16], dim=1)
        up32  = self.ups[3](up16) #64
        up256s = []
        for i in range(B):
            up64  = self.ups[4](up32[i:i+1]) #32
            up128 = self.ups[5](up64) #16
            up256 = self.ups[6](up128) #1
            up256s.append(up256)
        up256s = torch.cat(up256s, dim=0)
        #print(up256s.shape)

        return up256s

