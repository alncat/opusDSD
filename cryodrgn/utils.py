from datetime import datetime as dt
import os, sys
import numpy as np
import pickle
import collections
import functools
from torch import nn
import torch
from . import fft
import torch.nn.functional as F
from scipy import signal, ndimage
import healpy as hp
import math

_verbose = False
ALIGN_CORNERS = True

def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n, o] = data.shape
    mask = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1: # and k == 1:
                mask[i,j] = 1#/(1+r*26)
            else:
                mask[i,j] = r#/(1+r*26)
    print(data.shape)
    for i in range(m):
        for j in range(n):
            for k in range(o):
                result[:, :, i, j, k] = ndimage.convolve(data[:, :, i, j, k], mask, mode='wrap')
                #result[:, :, i, j, k] = signal.convolve2d(data[:, :, i, j, k], mask, boundary='symm', mode='same')
    return result

def ncc_loss(y_true, y_pred, win=None, ndims=2):
    """
    Local (over window) normalized cross correlation loss.
    """
    I = y_true
    J = y_pred

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    #ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross / torch.sqrt(I_var * J_var + 1e-5)

    return -torch.mean(cc)

def create_3dgrid(size, normalize=False):
    if normalize is False:
        zgrid, ygrid, xgrid = np.meshgrid(np.arange(size),
                                  np.arange(size),
                                  np.arange(size), indexing='ij')
    else:
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., size),
                                np.linspace(-1., 1., size),
                                np.linspace(-1., 1., size), indexing='ij')
    #return a grid of size (1, size, size, size, 3)
    return torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32))

def standardize_image(y, log=False):
    y_std = torch.std(y, (-1,-2), keepdim=True)
    y_mean = torch.mean(y, (-1,-2), keepdim=True)
    y = (y - y_mean)/y_std
    if log:
        print(y_mean, y_std)
    return y

def plot_image(axes, y_image, i, j, log=False, log_msg=""):
    y_image_std = np.std(y_image)
    y_image_mean = np.mean(y_image)
    if log:
        print(log_msg, y_image_mean, y_image_std)
    axes[i][j].imshow((y_image - y_image_mean)/y_image_std, cmap='gray')

def log(msg):
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    sys.stdout.flush()

def vlog(msg):
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()

def flog(msg, outfile):
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile,'a') as f:
            f.write(msg+'\n')
    except Exception as e:
        log(e)

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d) and m.weight.data.shape[-1] % 2 == 0:
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

def initseq(s, scale=1.):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

def crop_fft(x, out_size):
    in_size = x.size(-2)
    if in_size == out_size:
        return x
    assert in_size > out_size
    out_xdim = out_size//2 + 1
    head = (in_size - out_size)//2
    tail = head + out_size
    x = x[..., head:tail, :out_xdim]
    return x

def crop_image(x, out_size):
    in_size = x.size(-2)
    if in_size == out_size:
        return x
    assert in_size > out_size
    head = (in_size - out_size)//2
    tail = head + out_size
    x = x[..., head:tail, head:tail]
    return x

def downsample_image(img, down_size):
    full_size = img.shape[-1]
    img_fft = torch.fft.rfft2(img, dim=(-1,-2))
    img_fft = torch.fft.fftshift(img_fft, dim=(-2))
    img_fft = crop_fft(img_fft, down_size)*(down_size/full_size)**2
    img_fft = torch.fft.ifftshift(img_fft, dim=(-2))
    img = torch.fft.irfft2(img_fft, dim=(-1,-2))
    #print(img_fft.shape)
    return img

def mask_image_fft(img, mask, ctf_grid):
    img_fft = fft.torch_fft2_center(img)
    enc_circ_mask = ctf_grid.get_cos_mask(mask, mask+10)
    img_fft *= enc_circ_mask
    #img_fft = torch.fft.fftshift(img_fft, dim=(-2))
    #img_fft = img_fft[..., -mask-10+full_size:mask+10+full_size, :mask+11]
    #img_fft = torch.fft.ifftshift(img_fft, dim=(-2))
    #print(img_fft.shape)
    #img = fft.torch_ifft2_center(img_fft)
    return img_fft

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    kernel_input = (x - y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_xy(x, logstd, s=0.25):
    dim = x.size(1)
    gamma2 = dim*s
    var = torch.exp(2*logstd)
    invgamma2 = 1./(var + 1. + gamma2)
    #factor = 1. - (var+1.)*invgamma2*dim/2
    kernel = torch.exp(-(x.pow(2)*invgamma2).sum(-1)*0.5)*torch.sqrt(invgamma2.prod(-1))*gamma2**(dim//2)
    return kernel.mean()

def compute_xx(x, logstd, s=0.25):
    dim = x.size(1)
    gamma2 = dim*s
    xij = x.unsqueeze(1) - x.unsqueeze(0)
    var = torch.exp(2*logstd)
    varij = var.unsqueeze(1) + var.unsqueeze(0)
    invgamma2 = 1./(varij + gamma2)
    kernel = torch.exp(-(xij.pow(2)*invgamma2).sum(-1)*0.5)*torch.sqrt(invgamma2.prod(-1))*gamma2**(dim//2)
    return kernel.mean()

def compute_yy(x, logstd, s=0.25):
    dim = x.size(1)
    gamma2 = dim*s
    var = torch.exp(2*logstd)
    invgamma2 = 1./(2*var + gamma2)
    kernel = torch.sqrt(invgamma2.prod(-1))*gamma2**(dim//2)
    return kernel.mean()

def compute_cross_entropy(x, y, probs=None, s=0.25, eps=1e-3, neg=False):
    gamma = x.shape[-1]*s
    diff = x - y
    diff = diff.pow(2).sum(-1)
    if probs is None:
        probs = -diff.detach()*gamma*0.5
        probs = probs.exp()
    if neg:
        return (1 - probs)*torch.log(1 + 1./(diff+eps))
    else:
        return probs*torch.log(1 + diff)
    #logits = torch.log((1 - probs + eps)/(probs + eps))
    #return -logits*diff.exp()

def compute_smmd(x, logstd, s=0.25):
    xy = compute_xy(x, logstd, s=s)
    xx = compute_xx(x, logstd, s=s)
    kernel = -2.*xy + xx
    return kernel

def compute_cross_smmd(x, y, s=0.25, adaptive=False):
    dim = x.size(1) #(n, z)
    xyij = x.unsqueeze(-2) - y #(n, 1, z) - (n, B, z)
    if adaptive:
        xyijmedian = torch.median(xyij.pow(2).sum(-1), dim=-1, keepdim=True).values.detach() #(n)
        #print(xyijmedian)
        s = 0.5*xyijmedian/dim #(n)
    gamma2 = dim*s
    invgamma2 = 1./(0.1 + gamma2)
    kernel_xy = torch.exp(-(xyij.pow(2)).sum(-1)*0.5*invgamma2)*(invgamma2*gamma2)**(dim//2)

    xxij = x.unsqueeze(-2) - x.unsqueeze(-3) #(n, 1, z) - (1, n, z)
    invgamma2 = 1./(0.1 + gamma2)
    kernel_xx = torch.exp(-(xxij.pow(2)).sum(-1)*0.5*invgamma2)*(invgamma2*gamma2)**(dim//2)
    #print(kernel_xy.shape, kernel_xx.shape)

    kernel = -2.*kernel_xy.mean(-1) + kernel_xx.mean(-1)

    return kernel.mean()

def compute_cross_mmd(x, y, s=0.25, adaptive=False):
    dim = x.size(1)
    x = x.unsqueeze(0) #(1, n, z)
    xyij = x.unsqueeze(-2) - y.unsqueeze(-3) #(1, n, 1, z) - (B, 1, n, z)
    if adaptive:
        xyijmedian = torch.median(xyij.pow(2).sum(-1).view(-1)).detach()
        s = xyijmedian/dim
    gamma2 = dim*s
    invgamma2 = 1./(0.1 + gamma2)
    kernel_xy = torch.exp(-(xyij.pow(2)*invgamma2).sum(-1)*0.5)*(invgamma2*gamma2)**(dim//2)

    xxij = x.unsqueeze(-2) - x.unsqueeze(-3)
    invgamma2 = 1./(0.1 + gamma2)
    kernel_xx = torch.exp(-(xxij.pow(2)*invgamma2).sum(-1)*0.5)*(invgamma2*gamma2)**(dim//2)
    #print(kernel_xy.shape, kernel_xx.shape)

    kernel = -2.*kernel_xy + kernel_xx

    return kernel.mean()

def compute_image_mmd(y_recon, y_ref, s=0.25):
    B = y_recon.size(0)
    x = y_recon.view(B, -1)
    y = y_ref.view(B, -1)
    dim = x.size(1)
    gamma2 = dim*s
    xyij = x.unsqueeze(1) - y.unsqueeze(0)
    invgamma2 = 1./(2. + gamma2)
    kernel_xy = torch.exp(-(xyij.pow(2)*invgamma2).sum(-1)*0.5)*(invgamma2*gamma2)**(dim//2)

    xxij = x.unsqueeze(1) - x.unsqueeze(0)
    invgamma2 = 1./(2. + gamma2)
    kernel_xx = torch.exp(-(xxij.pow(2)*invgamma2).sum(-1)*0.5)*(invgamma2*gamma2)**(dim//2)

    kernel = -2.*kernel_xy + kernel_xx

    return kernel.mean()

def compute_cross_corr(mu):
    # normalize mu
    #mu = F.normalize(mu, dim=1)
    #print(mu, mu.pow(2).sum(-1))
    mu_mean = mu.mean(0)
    #mu_std = mu.std(0)
    mu = (mu - mu_mean)#/(mu_std + 1e-5)
    kernel = torch.matmul(mu[...,None], mu[...,None,:])
    #print(kernel.shape, torch.eye(mu.shape[-1]))
    #print(kernel.mean(0))
    kernel = kernel.mean(0) - torch.eye(mu.shape[-1]).to(mu.device)
    return kernel.pow(2).mean()

def compute_kld(mu, logstd):
    #mu2    = (torch.sum(mu**2 - 1., dim=-1)).abs()
    mu2 = torch.sum(mu**2, dim=-1)
    #mu2_c = torch.clip(mu2, min=4)
    #mu2_c = (mu2 - logstd.shape[-1]).abs()
    kernel = torch.sum(-logstd, dim=-1) + 0.5*(mu2 + torch.exp(2*logstd).sum(dim=-1))
    return kernel.mean(), mu2

def compute_kkld(x, logstd, s=0.25):
    xy = compute_xy(x, logstd, s=s)
    yy = compute_yy(x, logstd, s=s)
    kernel = -2.*xy + yy
    return kernel

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    #y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def compute_kldu(x, logstd, eps=1e-7):
    xij = x.unsqueeze(1) - x.unsqueeze(0)
    var = torch.exp(2*logstd)
    varij = var.unsqueeze(1) + var.unsqueeze(0)
    logstdij = torch.log(varij)*0.5
    xvar = (-xij.pow(2)/(varij + eps)).sum(-1)*0.5 - logstdij.sum(-1)
    kernel = torch.logsumexp(xvar, dim=-1)
    second_moment = x.pow(2).sum(-1) + var.sum(-1)
    kernel = kernel + second_moment*0.5
    return kernel.mean()

class SpatialTransformer2D(nn.Module):

    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        #vectors = [torch.arange(0, s) for s in size]

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        self.templateres = size
        self.normalize = normalize
        if self.normalize:
            ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
        else:
            ygrid, xgrid = np.meshgrid(np.arange(self.templateres),
                                  np.arange(self.templateres), indexing='ij')
        #xgrid is the innermost dimension (-1, ..., 1)
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid), axis=-1)[None].astype(np.float32)))

    def rotate(self, rot):
        return self.grid @ rot

    def translate(self, trans):
        pos = self.grid + 2.*trans/float(self.templateres - 1)
        return pos

    def sample(self, src, pos):
        return F.grid_sample(src, pos, align_corners=ALIGN_CORNERS)

    def pad(self, src, out_size):
        #pad the 2d output
        pad_size = (out_size - self.templateres)//2
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size))

    def rotate_and_sample(self, src, rot):
        pos = self.rotate(rot)
        return self.sample(src, pos)

    def translate_and_sample(self, src, trans):
        pos = self.translate(trans)
        return self.sample(src, pos)

    def get_2d_rot(self, rot_deg):
        theta = rot_deg*2.*np.pi
        B = rot_deg.shape[0]
        psi = torch.zeros(B, 2, 2)
        psi[:, 0, 0] = theta.cos()
        psi[:, 0, 1] = -theta.sin()
        psi[:, 1, 0] = theta.sin()
        psi[:, 1, 1] = theta.cos()
        return psi

    def rotate_translate_sample(self, src, rot_deg, trans):
        rot = self.get_2d_rot(rot_deg)
        pos = self.grid @ rot + 2.*trans/float(self.templateres - 1)
        return self.sample(src, pos)


def load_pkl(pkl):
    with open(pkl,'rb') as f:
        x = pickle.load(f)
    return x

def save_pkl(data, out_pkl, mode='wb'):
    if mode == 'wb' and os.path.exists(out_pkl):
        vlog(f'Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)

def R_from_eman(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def get_neighbor_rots(euler, order):
    euler_cpu = euler.cpu().numpy()
    neighbor_pix = hp.get_all_neighbours(order, euler_cpu[:, 1], euler_cpu[:, 0])
    print(neighbor_pix)
    #convert to euler angles
    neighbor_tilts, neighbor_psis = hp.pix2ang(order, neighbor_pix)
    print(neighbor_tilts.shape, neighbor_psis)
    neighbor_Rs = R_from_relion_bc(0, neighbor_tilts, neighbor_psis)
    print(neighbor_Rs)
    return neighbor_Rs

def R_from_relion_bc(a,b,y):
    #R = Ry Rb Ra
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(y), np.sin(y)
    cc = cb*ca
    cs = cb*sa
    sc = sb*ca
    ss = sb*sa
    R = np.zeros((b.shape[0], b.shape[1], 3, 3))
    R[:,:, 0,0] = cg*cc - sg*sa
    R[:,:, 1,0] = -sg*cc - cg*sa
    R[:,:, 2,0] = sc
    R[:,:, 0,1] = cg*cs + sg*ca
    R[:,:, 1,1] = -sg*cs + cg*ca
    R[:,:, 2,1] = ss
    R[:,:, 0,2] = -cg*sb
    R[:,:, 1,2] = sg*sb
    R[:,:, 2,2] = cb
    return R

def R_from_relion1(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(y), np.sin(y)
    cc = cb*ca
    cs = cb*sa
    sc = sb*ca
    ss = sb*sa
    #Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    #Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    #Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    #R = np.dot(np.dot(Ry,Rb),Ra)
    R = np.zeros((3, 3))
    R[0,0] = cg*cc - sg*sa
    R[0,1] = -sg*cc - cg*sa
    R[0,2] = sc
    R[1,0] = cg*cs + sg*ca
    R[1,1] = -sg*cs + cg*ca
    R[1,2] = ss
    R[2,0] = -cg*sb
    R[2,1] = sg*sb
    R[2,2] = cb
    return R

def R_from_relion_np(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,sy,0],[-sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    return R

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1,3)
    euler[:,0] += 90
    euler[:,2] -= 90
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix()*f
    return rot

def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3,3):
        rot = rot.reshape(1,3,3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    euler = RR.from_matrix(rot*f).as_euler('zxz', degrees=True)
    euler[:,0] -= 90
    euler[:,2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi/180
    return euler

def align_with_z(axis):
    #R = Ry Rb Ra
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    proj_mod = torch.sqrt(y**2 + z**2)
    R = torch.zeros(3, 3)#.to(axis.get_device())
    if proj_mod > 1e-5:
        R[..., 0, 0] = proj_mod
        R[..., 0, 1] = -x * y / proj_mod
        R[..., 0, 2] = -x * z / proj_mod
        R[..., 1, 0] = 0
        R[..., 1, 1] = z / proj_mod
        R[..., 1, 2] = -y / proj_mod
        R[..., 2, 0] = x
        R[..., 2, 1] = y
        R[..., 2, 2] = z
    else:
        R[..., 0, 2] = -1 if x > 0 else 1
        R[..., 1, 1] = 1
        R[..., 2, 0] = 1 if x > 0 else -1
    #print(R@axis)
    return R

def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[1.,0.,0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt

def torch_zrot(theta):
    psi = torch.zeros(3, 3).to(theta.get_device())
    psi[:, 0, 0] = theta.cos()
    psi[:, 0, 1] = theta.sin()
    psi[:, 1, 0] = -theta.sin()
    psi[:, 1, 1] = theta.cos()
    psi[:, 2, 2] = 1.
    return psi

def zrot(theta):
    #theta = psi#*2.*np.pi
    psi = np.zeros(3, 3)
    psi[0, 0] = np.cos(theta)
    psi[0, 1] = np.sin(theta)
    psi[1, 0] = -np.sin(theta)
    psi[1, 1] = np.cos(theta)
    psi[2, 2] = 1
    return psi

@memoized
def _zero_sphere_helper(D):
    xx = np.linspace(-1, 1, D, endpoint=True if D % 2 == 1 else False)
    z,y,x = np.meshgrid(xx,xx,xx)
    coords = np.stack((x,y,z),-1)
    r = np.sum(coords**2,axis=-1)**.5
    return np.where(r>1)

def zero_sphere(vol):
    '''Zero values of @vol outside the sphere'''
    assert len(set(vol.shape)) == 1, 'volume must be a cube'
    D = vol.shape[0]
    tmp = _zero_sphere_helper(D)
    vlog('Zeroing {} pixels'.format(len(tmp[0])))
    vol[tmp] = 0
    return vol

