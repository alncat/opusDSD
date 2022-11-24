import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import fft
from . import lie_tools
from . import utils

class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownBlock, self).__init__()

        down_block = []
        #down_block.append(nn.BatchNorm2d(inchannels))
        down_block.append(nn.Conv2d(inchannels, outchannels, 4, 2, 1))
        down_block.append(nn.LeakyReLU(0.1))
        down_block.append(nn.Conv2d(outchannels, outchannels, 3, 1, 1))
        down_block.append(nn.LeakyReLU(0.1))
        self.down = nn.Sequential(*down_block)
        #utils.initseq(self.down)

    def forward(self, x):
        return self.down(x)

class PoseEncoder(nn.Module):
    def __init__(self, image_size = 128, mode=None):

        super(PoseEncoder, self).__init__()
        self.start_size = image_size
        self.mode = mode
        self.down_out_dim = (self.start_size)//64
        self.zdim = 2

        self.down_outchannels = {2:512, 4:256, 8:128, 16:64, 32:32, 64:16, 128:8}
        self.init_conv = nn.Sequential(
                            nn.Conv2d(2, 8, 5, 1, 2),
                            nn.LeakyReLU(0.1)
                        )

        #utils.initseq(self.init_conv)
        downs = []
        resolution = 128
        self.n_down_layers = int(np.log2(resolution) - 1)
        inchannels = 8
        for i in range(self.n_down_layers):
            resolution //= 2
            outchannels = self.down_outchannels[resolution]
            down_module = DownBlock(inchannels, outchannels)
            downs.append(down_module)
            inchannels = outchannels

        self.downs = nn.ModuleList(downs)
        self.downz = nn.Sequential(
                nn.Linear(self.down_outchannels[2] * self.down_out_dim ** 2, 512), nn.LeakyReLU(0.1))

        if self.mode == 'rotation':
            self.pose = nn.Sequential(nn.Linear(512, self.zdim),
                                      nn.SoftMax(dim=-1))
        elif self.mode == 'shape':
            self.pose = nn.Sequential(nn.Linear(512, 1),
                                      nn.Sigmoid())
        else:
            self.pose = nn.Linear(512, self.zdim)

        #utils.initseq(self.downz)

        #utils.initmod(self.pose)
        self.xdim = 128//2
        x_idx = torch.arange(-self.xdim, self.xdim)/self.xdim #[-s, s]
        grid  = torch.meshgrid(x_idx, x_idx, indexing="ij")
        xgrid = grid[1] #change fast [[0,1,2,3]]
        ygrid = grid[0]
        grid = torch.stack([xgrid, ygrid], dim=-1)
        mask = grid.pow(2).sum(-1) < 0.85**2
        self.register_buffer("mask", mask)

    def forward(self, img):
        B = img.shape[0]
        img = img*self.mask
        x = self.init_conv(img)
        #print(x.shape)
        for i in range(self.n_down_layers):
            x = self.downs[i](x)
        z = self.downz(x.view(-1, self.down_outchannels[2]*self.down_out_dim ** 2))
        pose = self.pose(z)
        #trans = pose[:, 6:]
        return pose
