'''Lattice object'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import utils

log = utils.log
class CTFGrid:
    def __init__(self, D, device, center=False, vol_size=None):
        self.vol_size = D - 1
        assert self.vol_size >= 4, "volume size {} is smaller than 4".format(self.vol_size)
        self.x_size = self.vol_size//2 + 1
        x_idx = torch.arange(self.x_size).to(device)/float(self.vol_size) #(0, 0.5]
        if not center:
            y_idx = torch.arange(-self.x_size+2, self.x_size).to(device)/float(self.vol_size) #(-0.5, 0.5]
            grid  = torch.meshgrid(y_idx, x_idx, indexing='ij')
            grid_x = grid[1] #change fast [[0,1,2,3]]
            grid_y = torch.roll(grid[0], shifts=(self.x_size), dims=(0)) #fft shifted, center at the corner
            self.freqs2d = torch.stack((grid_x, grid_y), dim=-1)
        else:
            y_idx = torch.arange(-self.x_size+1, self.x_size-1).to(device)/float(self.vol_size)
            grid = torch.meshgrid(y_idx, x_idx, indexing='ij')
            self.freqs2d = torch.stack((grid[1], grid[0]), dim=-1)
        #print("ctf grid: ", self.freqs2d)
        self.D = D - 1
        self.device = device
        # flatten to 2d
        #self.freqs2d = self.freqs2d.view((-1,2))
        self.circular_masks = {}
        self.shells_index = torch.round(torch.sqrt(torch.sum((self.freqs2d * self.vol_size) ** 2, dim=-1)))
        self.shells_index = self.shells_index.type(torch.int64)
        self.max_r = self.shells_index.max()
        log("creating ctf grid {} with grid {}".format(center, self.shells_index))
        shells_index_val = self.shell_to_grid(torch.arange(self.vol_size).unsqueeze(0).to(device))
        print((shells_index_val - self.shells_index).sum())

        log("created ctf grid with shape: {}, max_r: {}".format(self.freqs2d.shape, self.max_r))
        #get number of frequencies per shell
        self.shells_weight = torch.ones_like(self.shells_index, dtype=torch.float32)

        self.shells_weight[:, 0] = 0.5
        if self.vol_size % 2 == 0:
            self.shells_weight[:, self.x_size - 1] = 0.5
        #self.shells_weight[0, 0] = 1.

        self.shells_count = self.get_shell_sum(self.shells_weight)
        #print(self.shells_count)
        #self.shells_count = self.shells_count.float()
        shell_count_validate = self.get_shell_count()
        assert torch.sum(self.shells_count.cpu() - shell_count_validate.cpu()).detach().cpu().numpy() == 0.
        self.s2 = self.freqs2d.pow(2).sum(-1)
        self.shells_s2 = self.get_shell_s2()

    def get_cos_mask(self, L, L_out):
        assert L < self.x_size, "L cannot be larger than {}".format(self.x_size)
        assert L < L_out, "L cannot be larger than {}".format(L_out)
        if L in self.circular_masks:
            return self.circular_masks[L]
        in_rad = L/self.vol_size
        out_rad = L_out/self.vol_size
        r = self.freqs2d.pow(2).sum(-1).sqrt()
        r = (r - in_rad)/(out_rad - in_rad)
        mask = torch.clip(r, min=0., max=1.)
        mask = torch.cos(mask*np.pi)*0.5 + 0.5
        self.circular_masks[L] = mask
        print(mask)
        return mask

    def get_circular_mask(self, L):
        assert L < self.x_size, "L cannot be larger than {}".format(self.x_size)
        if L in self.circular_masks:
            return self.circular_masks[L]
        mask = self.freqs2d.pow(2).sum(-1) < L*L/float(self.vol_size**2)
        mask = mask.float()
        print(mask)
        #mask = self.grid[..., 0]*self.grid[..., 0] + \
        #    self.grid[..., 1]*self.grid[..., 1] < L*L
        self.circular_masks[L] = mask
        return mask

    def get_shell_s2(self):
        radius = self.s2
        avg_radius = self.get_shell_sum(radius*self.shells_weight)/(self.shells_count + 1e-5)
        #avg_radius = torch.sqrt(avg_radius)
        return avg_radius

    def get_shell_sum(self, src):
        target = torch.zeros_like(src)

        #print(src.shape, self.shells_index.shape)
        if len(src.shape) == len(self.shells_index.shape) + 2:
            if src.is_cuda:
                batch_shells_index = self.shells_index.repeat((src.shape[0], src.shape[1], 1, 1)).to(src.get_device())
            else:
                batch_shells_index = self.shells_index.repeat((src.shape[0], src.shape[1], 1, 1))
            target_shells = torch.scatter_add(target, dim=-2, index=batch_shells_index, src=src)

        elif len(src.shape) == len(self.shells_index.shape) + 1:
            if src.is_cuda:
                batch_shells_index = self.shells_index.repeat((src.shape[0], 1, 1)).to(src.get_device())
            else:
                batch_shells_index = self.shells_index.repeat((src.shape[0], 1, 1))
            target_shells = torch.scatter_add(target, dim=-2, index=batch_shells_index, src=src)

        elif len(src.shape) == len(self.shells_index.shape):
            target_shells = torch.scatter_add(target, dim=-2, index=self.shells_index, src=src)
        else:
            log("shapes do not match {} {}".format(src.shape, self.shells_index.shape))
            raise RuntimeError
        target_shells = torch.sum(target_shells, dim=-1)
        return target_shells

    def get_shell_count(self):
        shell_count = torch.zeros((self.vol_size,)).to(self.device)
        for i in range(self.shells_index.shape[0]):
            for j in range(self.shells_index.shape[1]):
                shell_count[self.shells_index[i][j]] += self.shells_weight[i, j]
        return shell_count

    def shell_to_grid(self, x):
        B = x.shape[0]
        x_grid = x.unsqueeze(-1).repeat(1, 1, self.x_size)
        index = self.shells_index.repeat((B, 1, 1))
        x_grid = torch.gather(x_grid, 1, index)
        return x_grid

    def translation_grid(self, t):
        coords = self.freqs2d.to(t.get_device()) #(H, W, 2)
        #img = img # Bxhxw
        t = t.unsqueeze(-2).unsqueeze(-1) # Bxkx1x2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi # BxkxHxWx1
        tfilt = tfilt.squeeze(-1) # BxkxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # BxkxHxW
        s = torch.sin(tfilt) # BxkxHxN
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))
        return phase_shift

    def get_b_factor(self, b=1.):
        s2 = self.s2
        bfactor = torch.exp(-b/4*s2*4* np.pi**2)
        return bfactor

    def sample_local_translation(self, img, k, sigma):
        assert k > 0
        t = torch.randn((img.shape[0], k, 2))*sigma
        #t = torch.cat((torch.zeros(1, 2), t), dim=0)
        coords = self.freqs2d.to(img.get_device()) #(H, W, 2)
        #img = img # Bxhxw
        t = t.to(img.get_device())
        t = t.unsqueeze(-2).unsqueeze(-1) # BxCx1x2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi # BxCxHxWx1
        tfilt = tfilt.squeeze(-1) # BxCxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # BxHxW
        s = torch.sin(tfilt) # BxHxN
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))
        #print(t.shape, img.shape, phase_shift.shape)
        return img*phase_shift

    def translate_ft(self, img, t):
        '''
        Translate an image by phase shifting its Fourier transform

        Inputs:
            img: FT of image (B x img_dims x 2)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims x 2)

        img_dims can either be 2D or 1D (unraveled image)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs2d.to(img.get_device()) #(H, W, 2)
        # img = img # Bxhxw
        t = t.to(img.get_device())
        t = t.unsqueeze(-2).unsqueeze(-1) # BxCx1x2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi # BxCxHxWx1
        tfilt = tfilt.squeeze(-1) # BxCxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # BxHxW
        s = torch.sin(tfilt) # BxHxN
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))#.unsqueeze(1)
        #phase_shift = phase_shift.to(img.get_device())
        #print(t.shape, img.shape, phase_shift.shape)
        return (img*phase_shift)


class Grid:
    def __init__(self, D, device):
        print("initializing 2d grid of size ", D - 1)
        self.vol_size = D - 1
        self.x_size = self.vol_size//2 + 1
        #x_idx = torch.arange(0, self.vol_size, dtype=torch.float32)
        x_idx = torch.linspace(-1., 1., self.vol_size, dtype=torch.float32).to(device)
        grids = torch.meshgrid(x_idx, x_idx)
        #grids[1], (0, ..., vol)
        self.grid = torch.stack((grids[1], grids[0]), dim=-1)
        #print(self.grid)
        self.circular_masks = {}
        self.square_masks = {}

    def translate(self, images, trans):
        # trans (,2)
        #print(images.shape, self.grid.shape, trans.shape)
        #print(trans)
        #f, axes = plt.subplots(1, 2)
        #utils.plot_image(axes, images[0,...].detach().numpy(), 0)

        B = trans.shape[0]
        trans_dim = trans.shape[1]
        grid_t = self.grid.unsqueeze(0) - trans.view((B, 1, 1, trans_dim))
        # put in range -1, 1
        grid_t = 2.*(grid_t/float(self.vol_size - 1.) - 0.5)
        translated = F.grid_sample(images.unsqueeze(1), grid_t)
        #utils.plot_image(axes, translated[0,0,...].detach().numpy() - images[0,...].detach().numpy(), 1)
        #plt.show()
        return translated.squeeze(1)

    def get_square_mask(self, L):
        if L in self.square_masks:
            return self.square_masks[L]
        #L is the size of the square
        left = (self.vol_size - L)/2
        right = (self.vol_size - L)/2 + L

        mask_xl = self.grid[..., 0] >= left
        mask_xr = self.grid[..., 0] < right
        mask_x = mask_xl*mask_xr
        mask_yl = self.grid[..., 1] >= left
        mask_yr = self.grid[..., 1] < right
        mask_y = mask_yl*mask_yr
        mask = mask_x*mask_y
        self.square_masks[L] = mask
        return mask

    def get_circular_mask(self, in_rad, s=1., out_rad=0.95):
        in_rad *= s
        out_rad *= s
        L = int(in_rad*self.vol_size)
        if L in self.circular_masks:
            return self.circular_masks[L]
        r = self.grid.pow(2).sum(-1).sqrt()
        r = (r - in_rad)/(out_rad - in_rad)
        mask = torch.clip(r, min=0., max=1.)
        mask = torch.cos(mask*np.pi)*0.5 + 0.5
        self.circular_masks[L] = mask
        #print(mask)
        return mask


class Lattice:
    def __init__(self, D, extent=0.5, ignore_DC=True):
        assert D % 2 == 1, "Lattice size must be odd"
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, D, endpoint=True),
                             np.linspace(-extent, extent, D, endpoint=True))
        coords = np.stack([x0.ravel(),x1.ravel(),np.zeros(D**2)],1).astype(np.float32)
        self.coords = torch.tensor(coords)
        self.extent = extent
        self.D = D
        self.D2 = int(D/2)

        # todo: center should now just be 0,0; check Lattice.rotate...
        # c = 2/(D-1)*(D/2) -1
        # self.center = torch.tensor([c,c]) # pixel coordinate for img[D/2,D/2]
        self.center = torch.tensor([0.,0.])

        self.square_mask = {}
        self.circle_mask = {}

        self.freqs2d = self.coords[:,0:2]/extent/2 #(-0.5, 0.5)

        self.ignore_DC = ignore_DC

    def get_downsample_coords(self, d):
        assert d % 2 == 1
        extent = self.extent * (d-1) / (self.D-1)
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, d, endpoint=True),
                             np.linspace(-extent, extent, d, endpoint=True))
        coords = np.stack([x0.ravel(),x1.ravel(),np.zeros(d**2)],1).astype(np.float32)
        return torch.tensor(coords)

    def get_square_lattice(self, L):
        b,e = self.D2-L, self.D2+L+1
        center_lattice = self.coords.view(self.D,self.D,3)[b:e,b:e,:].contiguous().view(-1,3)
        return center_lattice

    def get_square_mask(self, L):
        '''Return a binary mask for self.coords which restricts coordinates to a centered square lattice'''
        if L in self.square_mask:
            return self.square_mask[L]
        assert 2*L+1 <= self.D, 'Mask with size {} too large for lattice with size {}'.format(L,self.D)
        log('Using square lattice of size {}x{}'.format(2*L+1,2*L+1))
        b,e = self.D2-L, self.D2+L
        c1 = self.coords.view(self.D,self.D,3)[b,b]
        c2 = self.coords.view(self.D,self.D,3)[e,e]
        m1 = self.coords[:,0] >= c1[0]
        m2 = self.coords[:,0] <= c2[0]
        m3 = self.coords[:,1] >= c1[1]
        m4 = self.coords[:,1] <= c2[1]
        mask = m1*m2*m3*m4
        self.square_mask[L] = mask
        if self.ignore_DC:
            raise NotImplementedError
        return mask

    def get_circular_mask(self, R):
        '''Return a binary mask for self.coords which restricts coordinates to a centered circular lattice'''
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert 2*R+1 <= self.D, 'Mask with radius {} too large for lattice with size {}'.format(R,self.D)
        log('Using circular lattice with radius {}'.format(R))
        r = R/(self.D//2)*self.extent
        mask = self.coords.pow(2).sum(-1) <= r**2
        if self.ignore_DC:
            assert self.coords[self.D**2//2].sum() == 0.0
            mask[self.D**2//2] = 0
        self.circle_mask[R] = mask
        return mask

    def rotate(self, images, theta):
        '''
        images: BxYxX
        theta: Q, in radians
        '''
        images = images.expand(len(theta), *images.shape) # QxBxYxX
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
        grid = self.coords[:,0:2]/self.extent @ rot # grid between -1 and 1
        grid = grid.view(len(rot), self.D, self.D, 2) # QxYxXx2
        offset = self.center - grid[:,self.D2,self.D2] # Qx2
        grid += offset[:,None,None,:]
        rotated = F.grid_sample(images, grid) # QxBxYxX
        return rotated.transpose(0,1) # BxQxYxX

    def translate_ft(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Fourier transform

        Inputs:
            img: FT of image (B x img_dims x 2)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims x 2)

        img_dims can either be 2D or 1D (unraveled image)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img.unsqueeze(1) # Bx1xNx2
        t = t.unsqueeze(-1) # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * -2 * np.pi # BxTxNx1
        tfilt = tfilt.squeeze(-1) # BxTxN
        c = torch.cos(tfilt) # BxTxN
        s = torch.sin(tfilt) # BxTxN
        return torch.stack([img[...,0]*c-img[...,1]*s,img[...,0]*s+img[...,1]*c],-1)

    def translate_ht(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Hartley transform

        Inputs:
            img: HT of image (B x img_dims)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims)

        img must be 1D unraveled image, symmetric around DC component
        '''
        #H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        center = int(len(coords)/2)
        img = img.unsqueeze(1) # Bx1xN
        t = t.unsqueeze(-1) # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi # BxTxNx1
        tfilt = tfilt.squeeze(-1) # BxTxN
        c = torch.cos(tfilt) # BxTxN
        s = torch.sin(tfilt) # BxTxN
        return c*img + s*img[:,:,np.arange(len(coords)-1,-1,-1)]


class EvenLattice(Lattice):
    '''For a DxD lattice where D is even, we set D/2,D/2 pixel to the center'''
    def __init__(self, D, extent=0.5, ignore_DC=False):
        # centered and scaled xy plane, values between -1 and 1
        # endpoint=False since FT is not symmetric around origin
        assert D % 2 == 0, "Lattice size must be even"
        if ignore_DC: raise NotImplementedError
        x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False),
                             np.linspace(-1, 1, D, endpoint=False))
        coords = np.stack([x0.ravel(),x1.ravel(),np.zeros(D**2)],1).astype(np.float32)
        self.coords = torch.tensor(coords)
        self.extent = extent
        self.D = D
        self.D2 = int(D/2)

        c = 2/(D-1)*(D/2) -1
        self.center = torch.tensor([c,c]) # pixel coordinate for img[D/2,D/2]

        self.square_mask = {}
        self.circle_mask = {}

        self.ignore_DC = ignore_DC

    def get_downsampled_coords(self, d):
        raise NotImplementedError
