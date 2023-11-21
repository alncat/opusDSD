'''Pytorch models'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import healpy as hp
from . import pose_encoder
from . import decoders
from . import fft
from . import lie_tools
from . import utils
from . import lattice
from . import mrc
from . import symm_groups
from . import unet
from . import healpix_sampler

log = utils.log
ALIGN_CORNERS = utils.ALIGN_CORNERS

class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(self, lattice, # Lattice object
            qlayers, qdim,
            players, pdim,
            in_dim, zdim = 1,
            encode_mode = 'resid',
            enc_mask = None,
            enc_type = 'linear_lowf',
            enc_dim = None,
            domain = 'fourier',
            activation = nn.ReLU,
            ref_vol = None,
            Apix = 1.,
            ctf_grid = None,
            template_type = None,
            warp_type = None,
            num_struct = 1,
            deform_emb_size = 2,
            device = None,
            symm = None,
            render_size=140,
            downfrac=0.5,
            templateres=192,
            tmp_prefix="ref",
            window_r=0.85):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        self.encode_mode = encode_mode
        self.num_struct = num_struct
        self.fixed_deform = False
        self.device = device
        self.render_size = (int((lattice.D - 1)*downfrac)//2)*2
        if ref_vol is not None:
            in_vol_nonzeros = torch.nonzero(ref_vol)
            in_vol_mins, _ = in_vol_nonzeros.min(dim=0)
            in_vol_maxs, _ = in_vol_nonzeros.max(dim=0)
            log("model: loading mask with nonzeros between {}, {}, {}".format(in_vol_mins, in_vol_maxs, ref_vol.shape))
            in_vol_maxs = ref_vol.shape[-1] - in_vol_maxs
            in_vol_min = min(in_vol_maxs.min(), in_vol_mins.min())
            mask_frac = (ref_vol.shape[-1] - in_vol_min*2 + 4) / ref_vol.shape[-1]
            log("model: masking volume using fraction: {}".format(mask_frac))
            self.window_r = min(mask_frac, 0.85)
            if templateres == 256:
                self.window_r = min(self.window_r, 0.85)
        else:
            self.window_r = window_r
        self.down_vol_size = int(self.render_size*self.window_r)//2*2
        self.encoder_crop_size = self.down_vol_size
        self.encoder_image_size = int(self.render_size*self.encoder_crop_size/self.down_vol_size)//2*2
        log("model: image supplemented into encoder will be of size {}".format(self.encoder_image_size))
        self.ref_vol = ref_vol

        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim*2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                            qlayers, # nlayers
                            qdim,  # hidden_dim
                            zdim*2, # out_dim
                            activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                            qlayers,
                            qdim, # hidden_dim
                            zdim*2, # out_dim
                            activation) #in_dim -> hidden_dim
        elif encode_mode == 'tilt':
            self.encoder = TiltEncoder(in_dim,
                            qlayers,
                            qdim,
                            zdim*2,
                            activation)
        elif encode_mode == 'fixed':
            #self.zdim = 256
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            self.pose_encoder = pose_encoder.PoseEncoder(image_size=128)
        elif encode_mode == 'deform':
            #self.zdim = 256
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            self.fixed_deform = True
        elif encode_mode == 'grad':
            self.encoder = Encoder(self.zdim, lattice.D, crop_vol_size=self.encoder_crop_size, in_mask=ref_vol, window_r=self.window_r)
            #self.shape_encoder = pose_encoder.PoseEncoder(image_size=128, mode="shape")
            self.fixed_deform = True
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        self.warp_type = warp_type

        self.encode_mode = encode_mode
        self.vanilla_dec = enc_type == "vanilla"
        self.template_type = template_type
        self.symm = symm
        self.deform_emb_size = deform_emb_size
        self.templateres = templateres
        self.decoder = get_decoder(3+zdim, lattice.D, players, pdim, domain, enc_type, enc_dim,
                                   activation, ref_vol=ref_vol, Apix=Apix,
                                   template_type=self.template_type, templateres=self.templateres,
                                   warp_type=self.warp_type,
                                   symm=self.symm, ctf_grid=ctf_grid,
                                   fixed_deform=self.fixed_deform, deform_emb_size=self.deform_emb_size,
                                   render_size=self.render_size, down_vol_size=self.down_vol_size, tmp_prefix=tmp_prefix)

    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'])
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D**2
        activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = HetOnlyVAE(lat,
                          c['qlayers'], c['qdim'],
                          c['players'], c['pdim'],
                          in_dim, c['zdim'],
                          encode_mode=c['encode_mode'],
                          enc_mask=enc_mask,
                          enc_type=c['pe_type'],
                          enc_dim=c['pe_dim'],
                          domain=c['domain'],
                          activation=activation)
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0],-1) for x in img)
        if self.enc_mask is not None:
            img = (x[:,self.enc_mask] for x in img)
        z = self.encoder(*img)
        #if self.encode_mode == 'fixed':
        #    z = torch.tile(self.encoder, (x.shape[0], 1))
        return z[:,:self.zdim], z[:,self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1]*(coords.ndimension()-2)), self.zdim)
        z = torch.cat((coords,z.expand(*coords.shape[:-1],self.zdim)),dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords,z))

    def get_fixedcode(self):
        return self.encoder()

    def vanilla_encode(self, img, rots=None, trans=None, eulers=None, num_gpus=4, snr2=1.):
        if self.encode_mode == 'fixed':
            z = self.encoder()
            z = z.repeat(num_gpus, 1)
            encout = {'encoding': None, 'z_mu': z}
        elif self.encode_mode == 'fixed_blur':
            #split encodings to template and blur kernel
            zs = self.encoder()
            z = zs[:1, :]
            #print(img.shape)
            encout = {"encoding": zs[1:, :]}
            #print(z.shape, encout['encoding'].shape)
        elif self.encode_mode == "grad":
            snr = np.sqrt(np.abs(snr2))
            #print(snr)
            encout = self.encoder(img, rots, trans, losslist=["kldiv"], eulers=eulers, snr=snr)
            mu     = encout["z_mu"]
            logstd = encout["z_logstd"]
            z      = encout["z"]
            #z  = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            encout["encoding"] = z
            x3d_center = encout["rotated_x"]
            #diff = (x3d_center.unsqueeze(1) - x3d_center.unsqueeze(0)).pow(2).sum(dim=(-1,-2))
            diff = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=(-1))
            top = torch.topk(diff, k=3, dim=-1, largest=False, sorted=True)
            #print(top.values)
            #print(top.indices[:, 1:], mu)
            encout["z_knn"] = mu[top.indices[:, 1:],:]
            #print(encout["z_knn"])
        return z, encout

    def vanilla_decode(self, rots, trans, z=None, save_mrc=False, eulers=None,
                       ref_fft=None, ctf=None, encout=None, others=None, mask=None):
        in_template = None
        if self.encode_mode != 'deform':
            #randomly perturb rotation
            #z = encout['encoding']
            #encout["z_mu"]     = mu
            #z, encout = self.vanilla_encode(img, rots, trans)
            pass
        else:
            #for deform embdding, the encoding will come from z
            encout = {'encoding': None}
        decout = self.decoder(rots, trans, z=z, in_template=in_template, save_mrc=save_mrc,
                              euler=eulers, ref_fft=ref_fft, ctf=ctf, others=others)
        decout["y_recon_ori"] = y_recon_ori = decout["y_recon"]
        #y_recon_ori = decout["y_recon"]*utils.crop_image(mask, self.down_vol_size)
        pad_size = (self.render_size - self.down_vol_size)//2
        #y_recon_ori = F.pad(y_recon_ori, (pad_size, pad_size, pad_size, pad_size))
        #if self.ref_vol is not None:
        decout["mask"] = F.pad(decout["mask"], (pad_size, pad_size, pad_size, pad_size))

        if len(ctf.shape) == 3:
            #print("ctf: ", ctf[0], others["ctf"].shape)
            ctf = ctf.unsqueeze(1)
        #print(y_recon_ori.shape, ctf.shape)
        #y_recon_ori  = torch.view_as_complex(y_recon_fft)
        #decout["y_recon_fft"] = y_recon_ori*ctf # ctf is (B, C, H, W) (B, 1, H, W, 2) x (B, 1, H, W, 1)
        #y_recon_fft   = decout["y_recon_fft"]
        #y_ref_fft   = torch.view_as_complex(decout["y_ref_fft"])
        #print(y_recon_fft.shape, y_ref_fft.shape, ctf.shape)
        # convert to image
        #y_recon_fft_s = torch.fft.fftshift(y_recon_fft, dim=(-2))
        #y_recon = fft.torch_ifft2_center(y_recon_fft_s)

        # put zero frequency on border
        #y_recon_fft_s = torch.fft.fftshift(y_recon_ori, dim=(-2))
        #y_recon_ori = fft.torch_ifft2_center(y_recon_fft_s)*mask

        ctf = torch.fft.fftshift(ctf, dim=(-2))
        ctf = utils.crop_fft(ctf, self.render_size)
        ctf = torch.fft.fftshift(ctf, dim=(-2))

        y_recon_fft = fft.torch_fft2_center(y_recon_ori)*ctf
        # put zero frequency in center
        #y_recon_fft = torch.fft.fftshift(y_recon_fft, dim=(-2))*ctf
        # put zero frequency on border
        #y_recon_fft = torch.fft.fftshift(y_recon_fft, dim=(-2))

        #y_ref_fft_s = torch.fft.fftshift(ref_fft, dim=(-2))
        #y_ref = fft.torch_ifft2_center(ref_fft)
        #print(y_ref.shape)

        decout["y_recon_fft"] = torch.view_as_real(y_recon_fft)
        #decout["y_ref_fft"] = torch.view_as_real(ref_fft)

        decout["y_recon"] = fft.torch_ifft2_center(y_recon_fft)
        #decout["y_recon_ori"] = y_recon_ori
        #decout["y_ref"] = y_ref
        # substract reconstruction to form diff image
        images = decout["y_recon"]
        rnd_factor = torch.rand([images.shape[0], 1, 1, 1], device=images.device)*0.2
        decout["y_recon_mean"] = images.detach().mean(dim=1, keepdim=True)
        #decout["y_ref"] -= decout["y_recon_mean"]*rnd_factor
        #decout["y_ref"] /= (1. - rnd_factor)

        return decout

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        if self.vanilla_dec:
            return self.vanilla_decode(*args, **kwargs)
        else:
            return self.decode(*args, **kwargs)

    def save_mrc(self, filename, enc=None, Apix=1.):
        if self.vanilla_dec:
            if enc is not None:
                self.decoder.save(filename, z=enc, Apix=Apix)

    def get_images(self, rots, trans):
        assert self.vanilla_dec
        return self.decoder.get_images(self.encoder(), rots, trans)

    def get_vol(self, z):
        if self.vanilla_dec:
            encoding = None
            if self.encode_mode == 'fixed':
                z = self.encoder()
            elif self.encode_mode == 'fixed_blur':
                z = self.encoder()
                #encout = self.affine_encoder(img)
                #encoding = encout['encoding']
                #z += encoding
            return self.decoder.get_vol(z=z)

def load_decoder(config, weights=None, device=None):
    '''
    Instantiate a decoder model from a config.pkl

    Inputs:
        config (str, dict): Path to config.pkl or loaded config.pkl
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    '''
    cfg = utils.load_pkl(config) if type(config) is str else config
    c = cfg['model_args']
    D = cfg['lattice_args']['D']
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
    model = get_decoder(3, D, c['layers'], c['dim'], c['domain'], c['pe_type'], c['pe_dim'], activation)
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model_state_dict'])
    if device is not None:
        model.to(device)
    return model

def get_decoder(in_dim, D, layers, dim, domain, enc_type, enc_dim=None, activation=nn.ReLU, templateres=128,
                ref_vol=None, Apix=1., template_type=None, warp_type=None,
                symm=None, ctf_grid=None, fixed_deform=False, deform_emb_size=2, render_size=140, down_vol_size=140, tmp_prefix="ref"):
    if enc_type == 'none':
        if domain == 'hartley':
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
            ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume # EW FIXME
        else:
            model = FTSliceDecoder(in_dim, D, layers, dim, activation)
        return model
    elif enc_type == 'vanilla':
        #model = VanillaDecoder
        #if template_type is None:
        #    assert ref_vol is not None
        return VanillaDecoder(D, ref_vol, Apix, template_type=template_type, templateres=templateres, warp_type=warp_type,
                              symm_group=symm, ctf_grid=ctf_grid,
                              fixed_deform=fixed_deform,
                              deform_emb_size=deform_emb_size,
                              zdim=in_dim - 3, render_size=render_size, down_vol_size=down_vol_size, tmp_prefix=tmp_prefix)
    else:
        model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder
        return model(in_dim, D, layers, dim, activation, enc_type=enc_type, enc_dim=enc_dim)

class FixedEncoder(nn.Module):
    def __init__(self, num_struct=1, in_dim=256):
        super(FixedEncoder, self).__init__()
        self.in_dim = in_dim
        self.num_struct=num_struct
        self.register_buffer('encoding1', torch.randn((self.num_struct, self.in_dim)))

    def forward(self,):
        return self.encoding1

class ConvTemplate(nn.Module):
    def __init__(self, in_dim=256, outchannels=1, templateres=128, affine=False):

        super(ConvTemplate, self).__init__()

        self.zdim = in_dim
        self.outchannels = outchannels
        self.templateres = templateres
        templateres = 256

        self.template1 = nn.Sequential(nn.Linear(self.zdim, 512), nn.LeakyReLU(0.2),
                                       nn.Linear(512, 2048), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 2048, 1024
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))

        inchannels, outchannels = 1024, 512
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))
        if self.templateres != templateres:
            self.template2 = nn.Sequential(*template2)

            inchannels, outchannels = 512, 256
            template3 = []
            template4 = []
            for i in range(int(np.log2(templateres)) - 3):
                if i < 2: #2:
                    template3.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                    template3.append(nn.LeakyReLU(0.2))
                else:
                    template4.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                    template4.append(nn.LeakyReLU(0.2))
                inchannels = outchannels
                outchannels = inchannels//2 #max(inchannels // 2, 16)
            self.template3 = nn.Sequential(*template3)
            self.template4 = nn.Sequential(*template4)
            for m in [self.template1, self.template2, self.template3, self.template4]:
                utils.initseq(m)
        else:
            inchannels, outchannels = 512, 256
            for i in range(int(np.log2(templateres)) - 3):
                template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                template2.append(nn.LeakyReLU(0.2))
                inchannels = outchannels
                outchannels = inchannels//2 #max(inchannels // 2, 16)
            self.template2 = nn.Sequential(*template2)
            self.template3 = self.template4 = []
            for m in [self.template1, self.template2]:
                utils.initseq(m)

        self.conv_out = nn.ConvTranspose3d(inchannels, 1, 4, 2, 1)
        #self.conv_out = nn.Conv3d(inchannels, 1, 3, 1, 1)

        utils.initmod(self.conv_out, gain=1./np.sqrt(templateres))

        self.intermediate_size = int(16*self.templateres/256)
        log('convtemplate: the output volume is of size {}, resample intermediate activations of size 16 to {}'.format(self.templateres, self.intermediate_size))

        # output rigid grid transformations

    def forward(self, encoding):
        #modules = [module for k, module in self.template2._modules.items()]
        #return checkpoint_sequential(modules, 2, self.template1(encoding).view(-1, 1024, 1, 1, 1))
        template2 = self.template2(self.template1(encoding).view(-1, 2048, 1, 1, 1))
        if self.templateres != 256:
            template3 = self.template3(template2) #(B, 64, 32, 32, 32)
            #can revise this to achieve other resolutions, current output of size 24*2^3
            template3 = F.interpolate(template3, size=self.intermediate_size, mode="trilinear", align_corners=ALIGN_CORNERS)
            template4 = self.template4(template3)
        else:
            template4 = template2

        return self.conv_out(template4), None

class AffineMixWeight(nn.Module):
    def __init__(self, in_dim=8, out_dim=3, out_size=32):
        super(AffineMixWeight, self).__init__()

        self.quat = utils.Quaternion()
        self.out_dim = out_dim

        inchannels = 8
        self.inchannels = inchannels
        self.warpf = nn.Sequential(
                nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, inchannels*2*2*2), nn.LeakyReLU(0.2)
                )
        outchannels = self.out_dim
        upsample = []
        n_layers = int(np.log2(out_size) - 1)
        for i in range(n_layers - 1):
            upsample.append(nn.ConvTranspose3d(inchannels, inchannels, 4, 2, 1))
            upsample.append(nn.LeakyReLU(0.2))
        upsample.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
        self.upsample = nn.Sequential(*upsample)

        utils.initseq(self.warpf)
        utils.initseq(self.upsample)

    def forward(self, encoding):
        init_vol = self.warpf(encoding).view(-1, self.inchannels, 2, 2, 2)
        out = self.upsample(init_vol)
        return out

class Encoder(nn.Module):
    def __init__(self, zdim, D, crop_vol_size, in_mask=None, window_r=None):
        super(Encoder, self).__init__()

        self.zdim = zdim
        self.inchannels = 1
        self.vol_size = D - 1
        self.crop_vol_size = crop_vol_size #int(160*self.scale_factor)
        self.window_r = window_r #(the cropping fraction of input mask)
        #downsample volume
        self.transformer_e = SpatialTransformer(self.crop_vol_size, render_size=self.crop_vol_size)
        #self.out_dim = (self.crop_vol_size)//128 + 1
        self.out_dim = 1
        log("encoder: the input image size is {}".format(self.crop_vol_size))
        #downsample mask
        if in_mask is not None:
            crop_mask_size = (int(in_mask.shape[-1]*self.window_r)//2)*2 #(self.crop_vol_size/128) previous default
            log("encoder: cropping mask from {} to {} using window {}".format(in_mask.shape, crop_mask_size, self.window_r))
            in_mask = self.transformer_e.crop(in_mask, crop_mask_size).unsqueeze(0).unsqueeze(0)
            # downsample
            in_mask = self.transformer_e.sample(in_mask)
            log("encoder: downsampling mask from {} to {}".format(crop_mask_size, in_mask.shape))
            self.register_buffer("in_mask", in_mask)
            self.use_mask = True
        else:
            self.register_buffer("in_mask", (self.transformer_e.grid.pow(2).sum(dim=-1) < 1).float())
            self.use_mask = True

        #self.init_conv = nn.Sequential(
        #                    nn.Conv2d(1, 8, 4, 2, 1),
        #                    nn.LeakyReLU(0.1)
        #                )

        downsample = []
        n_layers = int(np.log2(128//2))
        inchannels = 1
        outchannels = 32
        self.down2 = []
        downsample1 = []
        self.intermediate_size = 12
        for i in range(n_layers):
            if i < 3:
                downsample.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
                downsample.append(nn.LeakyReLU(0.2))
            else:
                downsample1.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
                downsample1.append(nn.LeakyReLU(0.2))
            inchannels = outchannels
            #if inchannels == outchannels:
            outchannels = min(inchannels * 2, 512)#1024)
            #else:
        self.out_channels = inchannels
        #downsample.append(nn.Conv3d(inchannels, self.out_channels, 4, 2, 1))
        #downsample.append(nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(*downsample)
        self.down2 = nn.Sequential(*downsample1)
        #downsample.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))

        #self.down1 = nn.Sequential(
        #        nn.Conv3d(self.inchannels, 16, 4, 2, 1),    nn.LeakyReLU(0.2),#40
        #        nn.Conv3d(16, 16, 4, 2, 1),   nn.LeakyReLU(0.2),#20
        #        nn.Conv3d(16, 32, 4, 2, 1),  nn.LeakyReLU(0.2),#10
        #        nn.Conv3d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),#5
        #        nn.Conv3d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),#2
        #        nn.Conv3d(64, self.out_channels, 4, 2, 1), nn.LeakyReLU(0.2))#1
        self.down3 = nn.Sequential(
                nn.Linear(self.out_channels * self.out_dim ** 3, 512), nn.LeakyReLU(0.2))

        self.mu = nn.Linear(512, self.zdim)
        self.logstd = nn.Linear(512, self.zdim)

        utils.initseq(self.down1)
        utils.initseq(self.down2)
        utils.initseq(self.down3)
        utils.initmod(self.mu)
        utils.initmod(self.logstd)

    def sample_neighbor_euler(self, coords, hp_order=32):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        euler_pixs = hp.ang2pix(hp_order//2, euler1, euler0, nest=True)

        neighbor_pix = np.random.randint(4, size=(coords.shape[0], 1)) + 4*euler_pixs[:, None]
        neighbor_pix = neighbor_pix.flatten()
        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order, neighbor_pix, nest=True)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(2) #(s*neighbor, 2)
        #print(neighbor_eulers.shape, neighbor_eulers_flatten.shape)
        return neighbor_eulers_flatten

    def forward(self, x, rots, trans, losslist=[], eulers=None, snr=1.):
        #2d to 3d suppose x is (N, 1, H, W)
        B = x.shape[0]
        #x = self.init_conv(x).unsqueeze(2)
        x = utils.crop_image(x, self.crop_vol_size).unsqueeze(2)
        # randomly scale and invert image
        #x = x*(0.75 + 0.5*torch.rand((B, 1, 1, 1, 1)).float().to(x.get_device())) \
        #    + 0.1*torch.randn((B, 1, 1, 1, 1)).to(x.get_device())

        pixrad = hp.max_pixrad(128)
        #x3d = x.repeat(1, 1, self.crop_vol_size, 1, 1) #(N, D, H, W)
        encs = []
        x3d_downs = []
        x3d_center = []
        for i in range(B):
            euler_i = eulers[i,...] #(B, 3)
            euler2 = eulers[i:i+1, 2] #(B)
            euler01 = euler_i[:2] #self.sample_neighbor_euler(eulers[i:i+1, ...])
            #rot = lie_tools.euler_to_SO3(sample_euler_i)#euler_i[...,:2])#.unsqueeze(1).unsqueeze(1) #(B, 1, 1, 3, 3)
            # convert to hopf
            #hopf = lie_tools.euler_to_hopf(euler_i)
            rot = lie_tools.hopf_to_SO3(euler01)

            # perturb z axis
            #rand_z = lie_tools.random_direction(1, pixrad*180/np.pi).to(o_rot.get_device())
            #rand_z = torch.transpose(o_rot, -2, -1) @ rand_z.unsqueeze(-1)
            #rand_z = rand_z.squeeze(-1)
            ## to radian
            #rand_e = lie_tools.direction_to_hopf(rand_z)
            ##new rotation
            #rot = lie_tools.hopf_to_SO3(rand_e.squeeze(0))

            #print(rand_e-euler01, torch.acos((torch.diag(o_rot.T @ rot).sum() - 1)/2)*180/np.pi)
            # minus z hopf angle == z euler angle
            euler2 = -euler2
            #pos = self.transformer_e.rotate(rot) # + 1)/2*(self.crop_vol_size - 1) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)
            #if self.use_mask:
            #    mask_2d = F.grid_sample(self.in_mask, pos, align_corners=ALIGN_CORNERS)
            #    mask_2d = (torch.sum(mask_2d, axis=-3) > 0.).squeeze(1)

            # if using hopf angles, use positive angle, if using euler, use negative
            x_i = self.transformer_e.rotate_2d(x[i], -euler2, mode='bicubic') #(1, 1, H, W)
            #print(x_i.shape, rot.T.shape)
            x3d_center.append(x_i.squeeze(1))#*mask_2d)

            if self.training:
                x_i = x_i*(1. + 0.3*snr*(torch.rand(1).float().to(x.get_device()) - 0.5)) \
                + 0.2*snr*torch.randn(1).to(x.get_device()) #+ 0.05*torch.rand_like(x_i)

            pos = self.transformer_e.rotate(rot.T) # + 1)/2*(self.crop_vol_size - 1) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)

            # construct pseudo volume
            x3d_i = x_i.unsqueeze(0).repeat(1, 1, self.crop_vol_size, 1, 1) #(N, C, D, H, W)

            # rotate the pseudo volume
            x3d_down = F.grid_sample(x3d_i, pos, align_corners=ALIGN_CORNERS)
            x3d_downs.append(x3d_down)

            #pos = self.transformer_e.rotate(rots[i].T)
            #x3d_downp = F.grid_sample(x3d[i:i+1], pos, align_corners=ALIGN_CORNERS)
            #print((x3d_down - x3d_downp).abs().mean()/((x3d_down).abs()+(x3d_downp).abs()).mean())
            #pass through convolution nn
        x3d_downs = torch.cat(x3d_downs, dim=0)
        x3d_center = torch.cat(x3d_center, dim=0)
        #print(x3d_downs.shape)
        # mask input
        if self.use_mask:
            x3d_downs *= self.in_mask
        # compute nearest neighbors in the same pose
        #x3d_center = x3d_downs[:, :, self.crop_vol_size//2, ...].squeeze(1).squeeze(1)

        enc1 = self.down1(x3d_downs)
        #print(enc1.shape, self.out_channels, self.out_dim)
        enc1 = F.interpolate(enc1, size=self.intermediate_size, mode="trilinear", align_corners=ALIGN_CORNERS) # 12^3
        enc2 = self.down2(enc1)
        encs = enc2.view(B, self.out_dim ** 3 *self.out_channels)
        encs = self.down3(encs)

        mu = self.mu(encs)
        if self.training:
            logstd = self.logstd(encs)
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            #z = mu
        else:
            logstd = torch.tensor(0.).to(mu.device)
            z = mu

        losses = {}
        if "kldiv" in losslist:
            #losses["kldiv"] = torch.mean(mu**2, dim=-1)
            losses["mu2"] = torch.sum(mu**2, dim=-1)
            losses["std2"] = torch.sum(torch.exp(2*logstd), dim=-1)
            #losses["kldiv"] = torch.mean(- logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
            #losses["kldiv"] = torch.sum(-logstd, dim=-1) + 0.5*losses["std2"] + 0.5*losses["mu2"]

        return {"z":z, "z_mu": mu, "losses": losses, "z_logstd": logstd, "rotated_x": x3d_center}


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, use_fourier=False, mode='bilinear', render_size=180):
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
        self.use_fourier = use_fourier
        self.render_size = render_size
        if self.normalize:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
        else:
            zgrid, ygrid, xgrid = np.meshgrid(np.arange(self.templateres),
                                  np.arange(self.templateres),
                                  np.arange(self.templateres), indexing='ij')
        #xgrid is the innermost dimension (-1, ..., 1)
        #self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))
        x_idx = torch.linspace(-1., 1., self.templateres) #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0)
        self.register_buffer("grid", grid)

        x_idx = torch.linspace(-1., 1., self.render_size) #[-s, s)
        grid  = torch.meshgrid(x_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        ygrid = grid[0]

        zgrid = torch.zeros_like(xgrid)
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("grid2d", grid)

    def rotate(self, rot):
        return self.grid @ rot #(1, 1, H, W, D, 3) @ (N, 1, 1, 1, 3, 3)

    def rotate_2d(self, ref, euler, out_size=None, mode='bicubic'):
        #euler (B,)
        rot_ref = lie_tools.zrot(euler).unsqueeze(1) #(B, 1, 3, 3)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        #print(self.grid2d.shape, rot_ref.shape)
        if out_size is not None:
            head = (self.render_size - out_size)//2
            tail = head + out_size
            grid_r = self.grid2d[..., head:tail, head:tail, :]
        else:
            grid_r = self.grid2d

        pos_ref = grid_r @ rot_ref
        rotated_ref = F.grid_sample(ref, pos_ref[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode=mode)
        return rotated_ref

    def rotate_euler(self, ref, euler):
        # ref (1, 1, z, y, x), euler (B, 2)
        Ra = lie_tools.zrot(euler[..., 0]).unsqueeze(1) #(B, 1, 3, 3)
        Rb = lie_tools.yrot(euler[..., 1]).unsqueeze(1)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, z, y, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        pos = self.gridz @ Ra
        # rotate around z, sample ref (1, z, y, x)
        rotated_ref = F.grid_sample(ref.squeeze(1), pos[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        #print(pos.shape, ref.shape, rotated_ref.shape)
        # permute y axis to z
        #print(rotated_ref.shape, ref.shape)
        rotated_ref = rotated_ref.permute(dims=[0, 2, 1, 3]) # (1, y, z, x)
        # sample ref
        pos = self.gridy @ Rb
        pos = torch.stack([pos[...,0], pos[...,2]], dim=-1)
        rotated_ref = F.grid_sample(rotated_ref, pos.squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        # permute again
        return rotated_ref#.permute(dims=[0, 2, 1, 3]) # (B, D, H, W)

    def sample(self, src):
        return F.grid_sample(src, self.grid, align_corners=ALIGN_CORNERS)

    def pad(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        pad_size = (out_size - src_size)//2
        if pad_size == 0:
            return src
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size))

    def crop(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        assert out_size <= src_size
        head = (src_size - out_size)//2
        tail = head + out_size
        if head == 0:
            return src
        return src[...,head:tail, head:tail, head:tail]

    def pad_FT(self, src, out_size):
        ypad_size = (out_size - self.render_size)//2
        return F.pad(src, (0, ypad_size, ypad_size, ypad_size-1))

    def rotate_and_sample(self, src, rot):
        pos = self.rotate(rot)
        return F.grid_sample(src, pos, align_corners=ALIGN_CORNERS)


    def forward(self, src, flow):
        # new locations
        # flow (N, 3, H, W, D)
        shape = flow.shape[2:]
        flow = flow.permute(0, 2, 3, 4, 1)
        new_locs = self.grid + flow

        # need to normalize grid values to [-1, 1] for resampler
        new_locs = 2. * (new_locs/float(self.templateres - 1) - 0.5)
        #for i in range(len(shape)):
        #    new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)

        return F.grid_sample(src, new_locs, align_corners=ALIGN_CORNERS, mode=self.mode)

class VanillaDecoder(nn.Module):
    def __init__(self, D, in_vol=None, Apix=1., template_type=None, templateres=256, warp_type=None, symm_group=None,
                 ctf_grid=None, fixed_deform=False, deform_emb_size=2, zdim=8, render_size=140,
                 use_fourier=False, down_vol_size=140, tmp_prefix="ref"):
        super(VanillaDecoder, self).__init__()
        self.D = D
        self.vol_size = (D - 1)
        self.Apix = Apix
        self.ctf_grid = ctf_grid
        self.template_type = template_type
        self.templateres = templateres
        self.use_conv_template = False
        self.fixed_deform = fixed_deform
        self.crop_vol_size = down_vol_size
        self.render_size = render_size
        self.use_fourier = use_fourier
        self.tmp_prefix = tmp_prefix

        if symm_group is not None:
            self.symm_group = symm_groups.SymmGroup(symm_group)
            print(self.symm_group.symm_opsR[self.symm_group.SymsNo - 1])
            self.register_buffer("symm_ops_rot", torch.tensor([x.rotation_matrix for x in self.symm_group.symm_opsR]).float())
            self.register_buffer("symm_ops_trans", torch.tensor([x.translation_vector for x in self.symm_group.symm_opsR]).float())
            grid_size = self.templateres
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size), indexing='ij')
            #xgrid is the innermost dimension (-1, ..., 1)
            self.register_buffer("symm_grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

        else:
            self.symm_group = None

        if self.template_type == "conv":
            self.use_conv_template = True
            self.zdim = zdim
            if self.use_fourier:
                self.template = ConvTemplate(in_dim=self.zdim, outchannels=1, templateres=self.templateres)
            else:
                self.template = ConvTemplate(in_dim=self.zdim, templateres=self.templateres, affine=False)
                x_idx = torch.linspace(-1., 1., 96) #[-s, s)
                grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
                xgrid = grid[2]
                ygrid = grid[1]
                zgrid = grid[0]
                grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0)
                self.register_buffer("grid_affine_weight", grid)
        else:
            self.template = nn.Parameter(in_vol)

        if self.use_fourier:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
            mask = torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32))
            mask = mask.pow(2).sum(-1) < 0.85 ** 2
            self.register_buffer("mask_w", mask)
            ##xgrid is the innermost dimension (-1, ..., 1)
            #self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

            self.fourier_transformer = healpix_sampler.SpatialTransformer(self.templateres, use_fourier=True, render_size=self.render_size)
        else:
            #crop_vol_size is the size of volume with density of interest, render_size is the size of output image after padding zeros
            #render_size also is equal to the size of downsampled experimental image
            self.transformer = SpatialTransformer(self.crop_vol_size, render_size=self.render_size)
            self.fourier_transformer = None #healpix_sampler.SpatialTransformer(self.crop_vol_size, use_fourier=True, render_size=self.crop_vol_size)
            if in_vol is not None:
                log("decoder: loading mask {}, volume render size is {}, volume of interest is {}".format(in_vol.shape, self.render_size, self.crop_vol_size))
                #resample input mask to render_size
                mask_frac = (self.crop_vol_size - 2)/self.render_size
                crop_size = int(in_vol.shape[-1]*mask_frac)//2*2
                crop_vol = self.transformer.crop(in_vol, crop_size).unsqueeze(0).unsqueeze(0)

                in_vol_nonzeros = torch.nonzero(crop_vol.squeeze())
                in_vol_mins, _ = in_vol_nonzeros.min(dim=0)
                in_vol_maxs, _ = in_vol_nonzeros.max(dim=0)
                log("decoder: cropped mask with nonzeros between {}, {}, {}".format(in_vol_mins, in_vol_maxs, crop_vol.shape))
                in_vol_maxs = crop_vol.shape[-1] - in_vol_maxs
                self.vol_bound = torch.minimum(in_vol_maxs, in_vol_mins).float()
                self.vol_bound *= (self.templateres/crop_vol.shape[-1]) #(templateres is the size of output volume)
                self.vol_bound = self.vol_bound.int() + 1
                log("decoder: setting volume boundary to {}".format(self.vol_bound))

                crop_vol = self.transformer.sample(crop_vol)
                log("decoder: cropping mask from {} to {}, cropping fraction is {}, downsample to {}".format(in_vol.shape[-1], crop_size, mask_frac, crop_vol.shape))
                self.register_buffer("ref_mask", crop_vol)

                apix_ori = self.Apix
                self.Apix = self.vol_size/self.render_size*self.Apix
                log("decoder: downsampling apix from {} to {}".format(apix_ori, self.Apix))
                #self.ref_mask_com = (self.transformer.grid*self.ref_mask.unsqueeze(-1)).mean(dim=(0, 1, 2, 3, 4))
                #print(self.ref_mask_com)
            else:
                apix_ori = self.Apix
                self.vol_bound = [1, 1, 1]
                self.Apix = self.vol_size/self.render_size*self.Apix
                log("decoder: downsampling apix from {} to {}".format(apix_ori, self.Apix))
                self.register_buffer("circle_mask", (torch.sum((torch.sum(self.transformer.grid ** 2, dim=-1) < 1.).float(), dim=-1) > 0.).float())
                self.ref_mask = None
                print(self.circle_mask.shape)

        self.warp_type = warp_type

    def symmetrise_template(self, template, grid):
        B = template.shape[0]
        symm_template = template
        for i in range(self.symm_group.SymsNo - 1):
            pos = grid @ self.symm_ops_rot[i] + self.symm_ops_trans[i]
            pos = pos.repeat(B,1,1,1,1)
            symm_template = symm_template + F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
        return symm_template/float(self.symm_group.SymsNo + 1)

    def sample_symmetrised_ops(self, rots):
        B = rots.size(0)
        rand_choices = torch.randint(self.symm_group.SymsNo, (B,))
        symm_rots  = self.symm_ops_rot[rand_choices]
        #symm_trans = self.symm_ops_trans[rand_choices]
        symm_rots  = symm_rots @ rots
        #symm_trans = self.symm_trans @ rots
        return symm_rots


    def affine_mixture(self, rot, t, weight):
        #print(weight.shape) #(64, 16, 16, 16)
        weight = F.softmax(weight, dim=0)
        #weight_32 = F.grid_sample(weight.unsqueeze(0), self.grid_affine_weight, align_corners=True).squeeze(0)
        #(8, D, H, W, 3)
        #print(self.grid_affine_weight.shape, t.shape, rot.shape, weight_32.shape)
        roted = (self.grid_affine_weight) @ rot.unsqueeze(1).unsqueeze(1) \
                    + t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #print(weight_32.shape, roted.shape)
        #print(rot_mean, t_mean - roted_t_mean)
        assert weight.shape == torch.Size([64, 16, 16, 16]) and roted.shape == torch.Size([64, 16, 16, 16, 3])
        grid = (weight.unsqueeze(-1)*roted).sum(dim=0, keepdim=True)
        return grid

    def get_particle_hopfs(self, coords, hp_order=64, depth=2):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        neighbor_pixs = hp.get_all_neighbours(hp_order//2, euler1, euler0, nest=True)

        neighbor_pixs = neighbor_pixs.flatten()
        neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
        neighbor_pixs = np.unique(neighbor_pixs)

        # sample again
        if depth == 1:
            neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            neighbor_pixs = hp.get_all_neighbours(hp_order//2, neighbor_euler1, neighbor_euler0, nest=True)
            # keep unique indices
            neighbor_pixs = neighbor_pixs.flatten()
            neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
            neighbor_pixs = np.unique(neighbor_pixs)
            # sample again
        if depth == 2:
            neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            neighbor_pixs = hp.get_all_neighbours(hp_order//2, neighbor_euler1, neighbor_euler0, nest=True)
            # keep unique indices
            neighbor_pixs = neighbor_pixs.flatten()
            neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
            neighbor_pixs = np.unique(neighbor_pixs)

        n_length = neighbor_pixs.shape[-1]

        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
        pixrad = hp.max_pixrad(hp_order//2)
        if depth == 2:
            n_sample = 50
        elif depth == 1:
            n_sample = 25
        else:
            n_sample = coords.shape[0]*8
        R = lie_tools.hopf_to_SO3(coords[:1, :].cpu())

        if n_length < n_sample:
            # convert to hopf
            rand_z = lie_tools.random_direction(n_sample - n_length, pixrad*4.*180/np.pi)
            rand_z = torch.transpose(R, -2, -1) @ rand_z.unsqueeze(-1)
            rand_z = rand_z.squeeze(-1)
            # to radian
            rand_e = lie_tools.direction_to_hopf(rand_z)*np.pi/180
            rand_angle0 = np.mod(rand_e[..., 0].cpu().numpy(), np.pi*2)
            rand_angle1 = rand_e[..., 1].cpu().numpy()
            #print(coords, rand_e*180/np.pi)

            #print(rand_angle0, rand_angle1, euler0, euler1)
            neighbor_euler0 = np.concatenate([neighbor_euler0, rand_angle0], axis=-1)
            neighbor_euler1 = np.concatenate([neighbor_euler1, rand_angle1], axis=-1)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(-1, 2) #(s*neighbor, 2)
        return neighbor_eulers_flatten

    def get_neighbor_hopfs(self, coords, hp_order=64):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        neighbor_pixs = hp.get_all_neighbours(hp_order//2, euler1, euler0, nest=True)
        #euler_pixs = hp.ang2pix(hp_order//2, euler1, euler0, nest=True)
        #neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs[:, None]

        neighbor_pixs = neighbor_pixs.flatten()
        neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
        n_length = neighbor_pixs.shape[-1]
        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)

        pixrad = hp.max_pixrad(hp_order//2)
        n_sample = 13
        R = lie_tools.hopf_to_SO3(coords.cpu())

        if n_length < n_sample:
            # convert to hopf
            rand_z = lie_tools.random_direction(n_sample - n_length, pixrad*4.*180/np.pi)
            rand_z = torch.transpose(R, -2, -1) @ rand_z.unsqueeze(-1)
            rand_z = rand_z.squeeze(-1)
            # to radian
            rand_e = lie_tools.direction_to_hopf(rand_z)*np.pi/180
            rand_angle0 = rand_e[..., 0].cpu().numpy()
            rand_angle1 = rand_e[..., 1].cpu().numpy()

            #euler_pixs_coarse = hp.ang2pix(hp_order//4, euler1, euler0, nest=True)
            #neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs_coarse[:, None]
            #neighbor_pixs = neighbor_pixs.flatten()
            #neighbor_euler1_c, neighbor_euler0_c = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            #neighbor_euler0_c = torch.tensor(neighbor_euler0_c).float().to(coords.get_device())/np.pi*180 #(s)
            #neighbor_euler1_c = torch.tensor(neighbor_euler1_c).float().to(coords.get_device())/np.pi*180

            neighbor_euler0 = np.concatenate([neighbor_euler0, rand_angle0], axis=-1)
            neighbor_euler1 = np.concatenate([neighbor_euler1, rand_angle1], axis=-1)
            neighbor_euler0 = np.concatenate([neighbor_euler0, euler0], axis=-1)
            neighbor_euler1 = np.concatenate([neighbor_euler1, euler1], axis=-1)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(-1, 2) #(s*neighbor, 2)
        return neighbor_eulers_flatten

    def get_neighbor_eulers(self, coords, hp_order=64):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        neighbor_pixs = hp.get_all_neighbours(hp_order//2, euler1, euler0, nest=True)
        #euler_pixs = hp.ang2pix(hp_order//2, euler1, euler0, nest=True)
        #neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs[:, None]

        neighbor_pixs = neighbor_pixs.flatten()
        neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
        #n_length = neighbor_pixs.shape[-1]
        #neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)

        neighbor_euler0 = euler0
        neighbor_euler1 = euler1
        n_length = 1

        pixrad = hp.max_pixrad(hp_order//2)
        n_sample = 12
        if n_length < n_sample:
            #rand_angle0 = (np.random.rand(n_sample - n_length) - 0.5)*pixrad + euler0
            #rand_angle1 = (np.random.rand(n_sample - n_length) - 0.5)*pixrad + euler1

            R = lie_tools.euler_to_SO3(coords.cpu())
            # convert to hopf
            hopf = lie_tools.euler_to_hopf(coords.cpu())
            #print(hopf, coords)
            R_h = lie_tools.hopf_to_SO3(hopf)
            #print(coords, hopf, lie_tools.hopf_to_euler(hopf))
            #print(R - R_h, R_h @ torch.transpose(R_h, -2, -1))
            hopf = lie_tools.direction_to_hopf(R[...,2,:])
            #print(hopf)
            #z = torch.tensor([0., 0., 1.])
            rand_z = lie_tools.random_direction(n_sample - n_length, pixrad*4.*180/np.pi)
            rand_z = torch.transpose(R, -2, -1) @ rand_z.unsqueeze(-1)
            rand_z = rand_z.squeeze(-1)
            #v = lie_tools.euler_to_direction(coords.cpu())
            #e = lie_tools.direction_to_euler(v)
            # to radian
            rand_e = lie_tools.direction_to_euler(rand_z)*np.pi/180
            rand_angle0 = rand_e[..., 0].cpu().numpy()
            rand_angle1 = rand_e[..., 1].cpu().numpy()
            #print((R @ v.unsqueeze(-1)).squeeze(-1))
            #print(rand_e - e, e - coords.cpu(), torch.acos((rand_z*v).sum(-1))/np.pi*180)
            #rand_so3 = lie_tools.random_biased_SO3(2, bias=20) @ R
            #rso3 = torch.transpose(rand_so3, -1, -2)
            #d = torch.sum(torch.diagonal(rso3 @ R, dim1=-2, dim2=-1), dim=-1)
            #print(d, torch.acos(d/2. - .5)*180/np.pi)
            #rand_e = lie_tools.so3_to_euler(rand_so3.cpu().numpy())
            #rand_angle0 = np.mod(rand_angle0, 2*np.pi)
            #rand_angle1 = np.mod(rand_angle1, np.pi)
            #euler_pixs_coarse = hp.ang2pix(hp_order//4, euler1, euler0, nest=True)
            #neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs_coarse[:, None]
            #neighbor_pixs = neighbor_pixs.flatten()
            #neighbor_euler1_c, neighbor_euler0_c = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            #neighbor_euler0_c = torch.tensor(neighbor_euler0_c).float().to(coords.get_device())/np.pi*180 #(s)
            #neighbor_euler1_c = torch.tensor(neighbor_euler1_c).float().to(coords.get_device())/np.pi*180

            #print(rand_angle0, rand_angle1, euler0, euler1)
            neighbor_euler0 = np.concatenate([neighbor_euler0, rand_angle0], axis=-1)
            neighbor_euler1 = np.concatenate([neighbor_euler1, rand_angle1], axis=-1)
            #neighbor_euler0 = np.concatenate([neighbor_euler0, euler0], axis=-1)
            #neighbor_euler1 = np.concatenate([neighbor_euler1, euler1], axis=-1)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        # sample euler at a coarser level
        #euler_pixs_coarse = hp.ang2pix(hp_order//4, euler1, euler0, nest=True)
        #neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs_coarse[:, None]
        #neighbor_pixs = neighbor_pixs.flatten()
        #neighbor_euler1_c, neighbor_euler0_c = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
        #neighbor_euler0_c = torch.tensor(neighbor_euler0_c).float().to(coords.get_device())/np.pi*180 #(s)
        #neighbor_euler1_c = torch.tensor(neighbor_euler1_c).float().to(coords.get_device())/np.pi*180

        #neighbor_euler0 = torch.cat([neighbor_euler0, neighbor_euler0_c], dim=0)
        #neighbor_euler1 = torch.cat([neighbor_euler1, neighbor_euler1_c], dim=0)

        #neighbor_euler0 = torch.cat([coords[:, 0].float(), neighbor_euler0], dim=-1)
        #neighbor_euler1 = torch.cat([coords[:, 1].float(), neighbor_euler1], dim=-1)

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #print(neighbor_eulers)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(-1, 2) #(s*neighbor, 2)
        return neighbor_eulers_flatten

    def lasso(self, template):
        head = self.vol_bound[2] #28
        tail  = template.shape[-1] - head + 1
        head_y = self.vol_bound[1] #28
        tail_y  = template.shape[-1] - head_y + 1
        head_z = self.vol_bound[0] #1
        tail_z  = template.shape[-1] - head_z + 1
        #print(head, head_y, head_z, tail, tail_y, tail_z)
        assert head >=1 and head_y >=1 and head_z >=1 and head < tail and head_y < tail_y and head_z < tail_z
        return (template[:, :, head_z:tail_z, head_y:tail_y, head:tail]).abs()

    def total_variation(self, template, vol_bound=None, sqrt=True):
        head = vol_bound[2] #28
        tail  = template.shape[-1] - head + 1
        head_y = vol_bound[1] #28
        tail_y  = template.shape[-1] - head_y + 1
        head_z = vol_bound[0] #1
        tail_z  = template.shape[-1] - head_z + 1
        assert head >=1 and head_y >=1 and head_z >=1 and head < tail and head_y < tail_y and head_z < tail_z
        out = (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z:tail_z, head_y:tail_y, head-1:tail-1])**2 + \
                  (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z:tail_z, head_y-1:tail_y-1, head:tail])**2 + \
                  (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z-1:tail_z-1, head_y:tail_y, head:tail])**2
        if sqrt:
            return torch.sqrt(1e-8 + out)
        else:
            return out

    def forward(self, rots, trans, z=None, in_template=None, euler=None, ref_fft=None, ctf=None,
                others=None, save_mrc=False, refine_pose=True, euler_particles=None,):
        if others is not None:
            others["y_fft"] = torch.view_as_complex(others["y_fft"])
        #ref_fft = torch.view_as_complex(ref_fft)
        #generate a projection
        # global rotation
        if self.use_conv_template:
            #print((z[0] == z[1]).sum())
            template, affine = self.template(z)
        elif in_template is not None:
            template = in_template
        else:
            template = self.template.unsqueeze(0).unsqueeze(0)

        losses = {}
        if self.training:
            losses["l2"] = torch.mean(self.lasso(template)).unsqueeze(0)
            losses["tvl2"] = torch.mean(self.total_variation(template, vol_bound=self.vol_bound)).unsqueeze(0) #torch.tensor(0.).to(template.get_device())
            if affine is not None:
                losses["tvl2"] = losses["tvl2"] + torch.mean(self.total_variation(affine[0], vol_bound=[1,1,1], sqrt=False)).unsqueeze(0)
        else:
            losses["l2"] = torch.tensor(0.).to(z.device)
            losses["tvl2"] = torch.tensor(0.).to(z.device)

        if self.symm_group is not None:
            #rots = self.sample_symmetrised_ops(rots)
            template = self.symmetrise_template(template, self.symm_grid)

        if self.use_fourier:
            #mask template
            #template = template * self.mask_w
            template_FT = fft.torch_rfft3_center(template)
            template_FT = template_FT[..., 1:, 1:, :self.templateres//2]
            template_FT = torch.cat((template_FT[..., 1:].flip(dims=(-1,-2,-3)).conj(), template_FT), dim=-1)
            #print(template_FT.shape)

        images = []
        refs = []
        masks = []
        euler_samples = []
        B = rots.shape[0]
        #theta = np.zeros((3,4), dtype=np.float32)

        if not refine_pose:
            mask_i = self.ref_mask.repeat(ref_fft.shape[1], 1, 1, 1, 1)
            if len(euler.shape) == 2:
                euler = euler.unsqueeze(1)
        for i in range(B):
            # compute affine transformation
            if affine is not None:
                affine_grid = torch.permute(self.grid_affine_weight, [0,4,1,2,3])
                #print(affine_grid)
                affine_grid = F.grid_sample(affine_grid, self.transformer.grid, align_corners=True)
                affine_grid = torch.permute(affine_grid, [0,2,3,4,1])
            #pos = F.affine_grid(torch.tensor(theta).unsqueeze(0), (1,1,self.vol_size,self.vol_size,self.vol_size))
            #pos = self.grid @ rots[i]#.transpose(-1, -2)

            if self.fixed_deform:
                if not self.use_fourier:
                    if refine_pose:
                        euler_i = euler[i:i+1,...] #(B, 3)
                        #R = lie_tools.euler_to_SO3(euler_i)
                        #euler_i = lie_tools.euler_to_hopf(euler_i)

                        #validate hopf
                        #Rh = lie_tools.hopf_to_SO3(euler_i[..., :2])
                        #Rz = lie_tools.hopf_to_SO3(euler_i[..., 2:])
                        #Rz1 = lie_tools.zrot(-euler_i[..., 2])
                        #print(R, Rz - Rz1, Rz @ Rh)

                        rand_ang = (torch.rand(1, 1).to(euler.get_device()) - .5)*360
                        #rand_ang = torch.zeros(1, 1).to(euler.get_device())
                        #rand_ang = euler_i[:, 2:]

                        euler2_sample_i = -euler_i[:, 2] #(1,) hopf angle == minus of euler angle
                        euler2 = euler2_sample_i + rand_ang.squeeze(1) # use minus if using euler

                        euler01 = euler_i[..., :2]
                        neighbor_eulers = self.get_particle_hopfs(euler01, hp_order=32, depth=0) #hp_order=64, depth=0)

                        len_euler = neighbor_eulers.shape[0]
                        n_eulers = torch.cat([neighbor_eulers, rand_ang.repeat(len_euler, 1)], dim=-1)
                        rot = lie_tools.hopf_to_SO3(n_eulers).unsqueeze(1).unsqueeze(1)

                        i_euler = torch.cat([euler_i[..., :2], rand_ang], dim=-1)
                        rot_i = lie_tools.hopf_to_SO3(i_euler).unsqueeze(1).unsqueeze(1)

                        #print(euler2.shape, neighbor_eulers.shape, rot.shape)
                        ref_i = ref_fft[i:i+1,...].repeat(euler2.shape[0], 1, 1, 1)
                        # convert neighbor_eulers to a list
                        euler_sample_i = neighbor_eulers#.repeat(2, 1)
                        euler2_sample_i = euler2_sample_i.unsqueeze(0).repeat(len_euler, 1).view(len_euler, -1)
                        euler_sample_i = torch.cat([euler_sample_i, -euler2_sample_i], dim=1)
                        #euler_samples.append(euler_sample_i)

                        pos = self.transformer.rotate(rot_i)
                        if self.ref_mask is not None:
                            valid = F.grid_sample(self.ref_mask, pos, align_corners=ALIGN_CORNERS)
                        else:
                            valid = None
                        if self.fourier_transformer is None:
                            template_i = template[i:i+1,...].repeat(rot.shape[0], 1, 1, 1, 1)
                            if affine is not None:
                                pos = affine_grid @ rot
                            else:
                                pos = self.transformer.rotate(rot)
                        else:
                            template_i = template[i:i+1,...]
                    else:
                        euler_i = euler[i,...] #(B, 3)
                        #rand_ang = (torch.rand(1, 1).to(euler.get_device()) - .5)*360
                        #euler2_sample_i = torch.cat([euler_i[:, 2]+0.4, eluer_i[:, 2], euler_i[:, 2]-0.4], dim=0)
                        euler2_sample_i = euler_i[:, 2] #(1,)
                        euler2 = euler2_sample_i
                        euler01 = euler_i[..., :2]
                        n_eulers = euler01 #torch.cat([euler01, ], dim=-1)
                        #rot_i = lie_tools.euler_to_SO3(euler01).unsqueeze(1).unsqueeze(1)
                        rot = lie_tools.euler_to_SO3(n_eulers).unsqueeze(1).unsqueeze(1)
                        #print(euler2.shape, neighbor_eulers.shape, rot.shape)
                        ref_i = ref_fft[i,...].unsqueeze(1)
                        template_i = template[i:i+1,...].repeat(rot.shape[0], 1, 1, 1, 1)
                        # convert neighbor_eulers to a list
                        euler_sample_i = euler_i#.repeat(2, 1)
                        #print(euler_sample_i, euler2_sample_i)
                        pos = self.transformer.rotate(rot)
                        valid = F.grid_sample(mask_i, pos, align_corners=ALIGN_CORNERS)
                    #else:
                    #    euler_i = euler[i:i+1,...] #(B, 3)
                    #    rand_ang = (torch.rand(1, 1).to(euler.device) - .5)*360
                    #    euler2 = euler_i[:, 2] - rand_ang.squeeze(1) #(B)
                    #    i_euler = torch.cat([euler_i[..., :2], rand_ang], dim=-1)
                    #    rot = lie_tools.euler_to_SO3(i_euler).unsqueeze(1).unsqueeze(1) #(B, 1, 1, 3, 3)
                    #    rot_i = rot
                    #    #Ra = lie_tools.zrot(euler_i[..., 0]).squeeze() #(B, 1, 3, 3)
                    #    #Rb = lie_tools.yrot(euler_i[..., 1]).squeeze()
                    #    #rot = Rb @ Ra
                    #    #print(euler_i.shape, euler2.shape, rot.shape)
                    #    #rot = rots[i].unsqueeze(0).unsqueeze(0).unsqueeze(0) #(1, 1, 1, 3, 3)
                    #    template_i = template[i:i+1,...]
                    #    ref_i = ref_fft[i:i+1, ...]
                    #    #for j in range(others_rot_i.shape[0]):
                    #    #    pos = self.transformer.rotate(others_rot_i[j])
                    #    #    image_j = F.grid_sample(template[i:i+1,...], pos, align_corners=ALIGN_CORNERS)
                    #    #    image_j *= valid
                    #    #    image.append(torch.sum(image_j, axis=-3).squeeze(0))

                    # rotate reference
                    ref = self.transformer.rotate_2d(ref_i, -euler2).squeeze(1)
                    refs.append(ref)
                    #pos = self.transformer.rotate(rot) # + 1)/2*(self.crop_vol_size - 1) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)
                    if valid is not None:
                        mask_i = (torch.sum(valid, axis=-3) > 0).detach().squeeze(1)
                    else:
                        mask_i = self.circle_mask
                    masks.append(mask_i)

                    # sample reference image from template
                    if self.fourier_transformer is None:
                        vol = F.grid_sample(template_i, pos, align_corners=ALIGN_CORNERS)
                        #vol = self.transformer.rotate_euler(template_i, euler_i)
                        #pos = self.transformer.rotate(rot_i)
                        #valid = F.grid_sample(self.ref_mask, pos, align_corners=ALIGN_CORNERS)
                        #vol = vol*valid
                        #image = torch.sum(vol, axis=-2)
                        image = torch.sum(vol, axis=-3).squeeze(1)
                        # Mask template using moving mask, generate projections
                    else:
                        # sample reference image from template using fourier transform
                        template_FT_i = self.fourier_transformer.fourier_transform(template_i)
                        rot = rot.squeeze(1)
                        image = self.fourier_transformer.rotate_and_sampleFT(template_FT_i, rot)
                        image = fft.torch_ifft2_center(image).squeeze(0)
                        # further pad image
                        image = self.transformer.pad(image, self.render_size)

                        # we can further search poses here
                        if False:
                            # crop ctf
                            image_ctf = self.fourier_transformer.ctf_correction(image, ctf[i:i+1, ...])
                            mask_i = self.transformer.pad(mask_i, self.render_size)
                            masked_ref = fft.torch_fft2_center(mask_i*ref)
                            top_angles = self.fourier_transformer.find_top_angles(image_ctf.detach(), masked_ref.detach(), euler_sample_i)

                            # search euler angles one more time
                            euler01 = top_angles[..., :2]
                            neighbor_eulers = self.get_particle_hopfs(euler01, hp_order=64, depth=0)

                            len_euler = neighbor_eulers.shape[0]
                            n_eulers = torch.cat([neighbor_eulers, rand_ang.repeat(len_euler, 1)], dim=-1)
                            rot = lie_tools.hopf_to_SO3(n_eulers).unsqueeze(1).unsqueeze(1)

                            rot = rot.squeeze(1)
                            image = self.fourier_transformer.rotate_and_sampleFT(template_FT_i, rot)
                            image = fft.torch_ifft2_center(image).squeeze(0)

                            # convert neighbor_eulers to a list
                            euler_sample_i = neighbor_eulers
                            euler2_sample_i = -euler_i[:, 2]
                            euler2_sample_i = euler2_sample_i.unsqueeze(0).repeat(len_euler, 1).view(len_euler, -1)
                            euler_sample_i = torch.cat([euler_sample_i, -euler2_sample_i], dim=1)

                    # append sampled angles
                    euler_samples.append(euler_sample_i)

                else:
                    #image = self.fourier_transformer.rotate_and_sampleFT(template_FT[i:i+1,...],
                    #                                                     rots[i], image_ori).squeeze(0)

                    #image, _ = self.fourier_transformer.rotate_and_sample_euler(template_FT[i:i+1,...],
                    #                                                            euler[i:i+1,...], ref_fft[i:i+1,...])
                    #image, ref = self.fourier_transformer.hp_sample_(template_FT[i:i+1,...],
                    #                                           euler, ref_fft, ctf,
                    #                                           trans=trans)
                    if others is not None:
                        # concatenate ref, euler, ctf, trans with others
                        euler_i = torch.cat([euler[i:i+1,...], others["euler"][i,...]], dim=0)
                        trans_i = torch.cat([trans[i:i+1,...], others["trans"][i,...]], dim=0)
                        #print(euler_i.shape, others["euler"].shape, trans_i.shape, ref_fft.shape, others["y_fft"].shape)
                        ref_fft_i = torch.cat([ref_fft[i,...], others["y_fft"][i,...]], dim=0)
                    else:
                        euler_i = euler[i:i+1,...]
                        trans_i = trans[i:i+1,...]
                        ref_fft_i = ref_fft[i,...]
                    image, ref = self.fourier_transformer.hp_sample_(template_FT[i:i+1,...],
                                                               euler_i, ref_fft_i, ctf[i:i+1,...],
                                                               trans=trans_i)

                    image = image.squeeze(0)
                    ref = ref.squeeze(0)
                    refs.append(ref)

            elif self.warp_type == "affine":
                if self.use_fourier:
                    image, ref = self.fourier_transformer.hp_sample_(template_FT,
                                                               euler[i:i+1,...], ref_fft[i:i+1,...], ctf[i:i+1,...],
                                                               trans=trans[i:i+1,...])
                    image = image.squeeze(0) #(1, B, H, W)
                    ref = ref.squeeze(0) #(1, B, H, W)
                    refs.append(ref)
            else:
                if refine_pose:
                    euler_i = euler[i:i+1,...] #(B, 3)

                    rand_ang = (torch.rand(1, 1).to(euler.get_device()) - .5)*360

                    euler2_sample_i = -euler_i[:, 2] #(1,) hopf angle == minus of euler angle
                    euler2 = euler2_sample_i + rand_ang.squeeze(1) # use minus if using euler

                    euler01 = euler_i[..., :2]
                    neighbor_eulers = self.get_particle_hopfs(euler01, hp_order=32, depth=0) #hp_order=64, depth=0)

                    len_euler = neighbor_eulers.shape[0]
                    n_eulers = torch.cat([neighbor_eulers, rand_ang.repeat(len_euler, 1)], dim=-1)
                    rot = lie_tools.hopf_to_SO3(n_eulers).unsqueeze(1).unsqueeze(1)

                    i_euler = torch.cat([euler_i[..., :2], rand_ang], dim=-1)
                    rot_i = lie_tools.hopf_to_SO3(i_euler).unsqueeze(1).unsqueeze(1)

                    ref_i = ref_fft[i:i+1,...].repeat(euler2.shape[0], 1, 1, 1)
                    # convert neighbor_eulers to a list
                    euler_sample_i = neighbor_eulers#.repeat(2, 1)
                    euler2_sample_i = euler2_sample_i.unsqueeze(0).repeat(len_euler, 1).view(len_euler, -1)
                    euler_sample_i = torch.cat([euler_sample_i, -euler2_sample_i], dim=1)
                    pos = self.transformer.rotate(rot_i)
                    valid = F.grid_sample(self.ref_mask, pos, align_corners=ALIGN_CORNERS)
                    template_i = template.repeat(rot.shape[0], 1, 1, 1, 1)
                    if affine is not None:
                        pos = affine_grid @ rot
                    else:
                        pos = self.transformer.rotate(rot)
                    vol = F.grid_sample(template_i, pos, align_corners=ALIGN_CORNERS)
                    image = torch.sum(vol, axis=-3).squeeze(1)
                    # rotate reference
                    ref = self.transformer.rotate_2d(ref_i, -euler2).squeeze(1)
                    refs.append(ref)
                    mask_i = (torch.sum(valid, axis=-3) > 0).detach().squeeze(1)
                    masks.append(mask_i)
                else:
                    raise RuntimeError
            images.append(image)
        images = torch.stack(images, 0)
        refs   = torch.stack(refs, 0)
        masks  = torch.stack(masks, 0)
        if len(euler_samples):
            euler_samples = torch.stack(euler_samples, 0)
        # pad to original size
        if not self.use_fourier:
            images = self.transformer.pad(images, self.render_size)
        if save_mrc:
            if self.use_fourier:
                self.save_mrc(template_FT[0:1, ...], self.tmp_prefix, flip=False)
            else:
                self.save_mrc(template[0:1, ...], self.tmp_prefix, flip=False)
        return {"y_recon": images, "losses": losses, "y_ref": refs, "mask": masks, "euler_samples": euler_samples}

    def save_mrc(self, template, filename, flip=False):
        with torch.no_grad():
            dev_id = template.get_device()
            if self.use_fourier:
                #template_FT = fft.torch_rfft3_center(template)
                #the origin is at self.templateres//2 - 1
                start = (self.templateres - self.vol_size)//2 - 1
                template_FT = template[..., start:start+self.vol_size, start:start+self.vol_size, \
                                      self.templateres//2-1:self.templateres//2+self.vol_size//2]
                template_FT = template_FT*(self.vol_size/self.templateres)**3
                #print(template_FT.shape)
                template = fft.torch_irfft3_center(template_FT)
            elif self.transformer.templateres != self.vol_size:
                template = self.transformer.sample(template)
            template = template.squeeze(0).squeeze(0)
            if flip:
                template = template.flip(0)
            mrc.write(filename + str(dev_id) + ".mrc", template.detach().cpu().numpy(), Apix=self.Apix, is_vol=True)

    @torch.no_grad()
    def save(self, filename, z=None, encoding=None, flip=False, Apix=1.):
        if self.template_type == "conv":
            template, _ = self.template(z)
            if self.transformer.templateres != self.templateres:
                #resample
                #template = F.grid_sample(template, self.grid, align_corners=True)
                template = self.transformer.sample(template)
            template = template.squeeze(0).squeeze(0)
        else:
            template = self.template
        if flip:
            template = template.flip(0)
        mrc.write(filename + ".mrc", template.detach().cpu().numpy(), Apix=Apix, is_vol=True)

    def get_vol(self, z=None):
        if self.template_type == "conv":
            template, _ = self.template(z)
            if self.transformer.templateres != self.vol_size:
                #resample
                template = self.transformer.sample(template)
        else:
            template = self.template
        return template

    def get_images(self, z, rots, trans):
        if self.template_type == "conv":
            template, _ = self.template(z)
        else:
            template = self.template
        B = rots.shape[0]
        images = []
        for i in range(B):
            pos = self.transformer.rotate(rots[i])
            valid = (torch.sum(pos ** 2, dim=-1) < 1.).float()

            vol = F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
            vol *= valid
            image = torch.sum(vol, axis=-3)
            image = image.squeeze(0)
            images.append(image)
        images = torch.stack(images, 0)
        if self.transformer.templateres != self.vol_size:
            images = self.transformer.pad(images, self.vol_size)
        images = self.translate(images, trans)
        return images


class PositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[...,0:3].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[0:-1,0:-1,0:-1]) # remove last +k freq for inverse FFT
        return vol

class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2]//2 # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c # include the origin
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:cc,:])
        image[..., 0:cc] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,cc:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[...,0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            keep = x.pow(2).sum(dim=1) <= extent**2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x,z.expand(x.shape[0],zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[...,0] - y[...,1]
                slice_ = torch.zeros(D**2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D,D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class FTSliceDecoder(nn.Module):
    '''
    Evaluate a central slice out of a 3D FT of a model, returns representation in
    Hartley reciprocal space

    Exploits the symmetry of the FT where F*(x,y) = F(-x,-y) and only
    evaluates half of the lattice. The decoder is f(x,y,z) => real, imag
    '''
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation):
        '''D: image width or height'''
        super(FTSliceDecoder, self).__init__()
        self.decoder = ResidLinearMLP(in_dim, nlayers, hidden_dim, 2, activation)
        D2 = int(D/2)

        ### various pixel indices to keep track of for forward_even
        self.center = D2*D + D2
        self.extra = np.arange((D2+1)*D, D**2, D) # bottom-left column without conjugate pair
        # evalute the top half of the image up through the center pixel
        # and extra bottom-left column (todo: just evaluate a D-1 x D-1 image so
        # we don't have to worry about this)
        self.all_eval = np.concatenate((np.arange(self.center+1), self.extra))

        # pixel indices for the top half of the image up to (but not incl)
        # the center pixel and excluding the top row and left-most column
        i, j = np.meshgrid(np.arange(1,D),np.arange(1,D2+1))
        self.top = (j*D+i).ravel()[:-D2]

        # pixel indices for bottom half of the image after the center pixel
        # excluding left-most column and given in reverse order
        i, j =np.meshgrid(np.arange(1,D),np.arange(D2,D))
        self.bottom_rev = (j*D+i).ravel()[D2:][::-1].copy()

        self.D = D
        self.D2 = D2

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        assert lattice.shape[-2] % 2 == 1
        c = lattice.shape[-2]//2 # center pixel
        assert lattice[...,c,0:3].sum() == 0.0, '{} != 0.0'.format(lattice[...,c,0:3].sum())
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:c+1,:])
        image[..., 0:c+1] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,c+1:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def forward_even(self, lattice):
        '''Extra bookkeeping with extra row/column for an even sized DFT'''
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,self.all_eval,:])
        image[..., self.all_eval] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,self.bottom_rev] = top_half[...,self.top,0] + top_half[...,self.top,1]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(lattice)
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.decode(x)
                y = y[...,0] - y[...,1]
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class VAE(nn.Module):
    def __init__(self,
            lattice,
            qlayers, qdim,
            players, pdim,
            encode_mode = 'mlp',
            no_trans = False,
            enc_mask = None
            ):
        super(VAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 2
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, qdim)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(self.in_dim,
                            qlayers-2, # -2 bc we add 2 more layers in the homeomorphic encoer
                            qdim,  # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        elif encode_mode == 'mlp':
            self.encoder = MLP(self.in_dim,
                            qlayers-2,
                            qdim, # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        # predict rotation and translation in two completely separate NNs
        #self.so3_encoder = SO3reparameterize(qdim) # hidden_dim -> SO(3) latent variable
        #self.trans_encoder = ResidLinearMLP(nx*ny, 5, qdim, 4, nn.ReLU)

        # or predict rotation/translations from intermediate encoding
        self.so3_encoder = SO3reparameterize(qdim, 1, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(qdim, 1, qdim, 4, nn.ReLU)

        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, img):
        '''img: BxDxD'''
        img = img.view(img.size(0),-1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
        enc = nn.ReLU()(self.encoder(img))
        z_mu, z_std = self.so3_encoder(enc)
        if self.no_trans:
            tmu, tlogvar = (None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
        return z_mu, z_std, tmu, tlogvar

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def decode(self, rot):
        # transform lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)
        return y_hat

    def forward(self, img):
        z_mu, z_std, tmu, tlogvar = self.encode(img)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        # transform lattice by rot and predict image
        y_hat = self.decode(rot)
        if not self.no_trans:
            # translate image by t
            B = img.size(0)
            t = self.reparameterize(tmu, tlogvar)
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), t)
            img = img.view(B,self.D, self.D)
        return y_hat, img, z_mu, z_std, w_eps, tmu, tlogvar

class TiltVAE(nn.Module):
    def __init__(self,
            lattice, tilt,
            qlayers, qdim,
            players, pdim,
            no_trans=False,
            enc_mask=None
            ):
        super(TiltVAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 3
        self.encoder = ResidLinearMLP(self.in_dim,
                                      qlayers-3,
                                      qdim,
                                      qdim,
                                      nn.ReLU)
        self.so3_encoder = SO3reparameterize(2*qdim, 3, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(2*qdim, 2, qdim, 4, nn.ReLU)
        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        assert tilt.shape == (3,3), 'Rotation matrix input required'
        self.tilt = torch.tensor(tilt)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def encode(self, img, img_tilt):
        img = img.view(img.size(0), -1)
        img_tilt = img_tilt.view(img_tilt.size(0), -1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
            img_tilt = img_tilt[:,self.enc_mask]
        enc1 = self.encoder(img)
        enc2 = self.encoder(img_tilt)
        enc = torch.cat((enc1,enc2), -1) # then nn.ReLU?
        z_mu, z_std = self.so3_encoder(enc)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        if self.no_trans:
            tmu, tlogvar, t = (None,None,None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
            t = self.reparameterize(tmu, tlogvar)
        return z_mu, z_std, w_eps, rot, tmu, tlogvar, t

    def forward(self, img, img_tilt):
        B = img.size(0)
        z_mu, z_std, w_eps, rot, tmu, tlogvar, t = self.encode(img, img_tilt)
        if not self.no_trans:
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), -t)
            img_tilt = self.lattice.translate_ht(img_tilt.view(B,-1), -t)
            img = img.view(B, self.D, self.D)
            img_tilt = img_tilt.view(B, self.D, self.D)

        # rotate lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)

        # tilt series pair
        x = self.lattice.coords @ self.tilt @ rot
        y_hat2 = self.decoder(x)
        y_hat2 = y_hat2.view(-1, self.D, self.D)
        return y_hat, y_hat2, img, img_tilt, z_mu, z_std, w_eps, tmu, tlogvar

# fixme: this is half-deprecated (not used in TiltVAE, but still used in tilt BNB)
class TiltEncoder(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(TiltEncoder, self).__init__()
        assert nlayers > 2
        self.encoder1 = ResidLinearMLP(in_dim, nlayers-2, hidden_dim, hidden_dim, activation)
        self.encoder2 = ResidLinearMLP(hidden_dim*2, 2, hidden_dim, out_dim, activation)
        self.in_dim = in_dim

    def forward(self, x, x_tilt):
        x_enc = self.encoder1(x)
        x_tilt_enc = self.encoder1(x_tilt)
        z = self.encoder2(torch.cat((x_enc,x_tilt_enc),-1))
        return z

class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        #self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )
    def forward(self, x):
        x = x.view(-1,1,64,64)
        x = self.main(x)
        return x.view(x.size(0), -1) # flatten

class SO3reparameterize(nn.Module):
    '''Reparameterize R^N encoder output to SO(3) latent variable'''
    def __init__(self, input_dims, nlayers=None, hidden_dim=None):
        super().__init__()
        if nlayers is not None:
            self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
        else:
            self.main = nn.Linear(input_dims, 9)

        # start with big outputs
        #self.s2s2map.weight.data.uniform_(-5,5)
        #self.s2s2map.bias.data.uniform_(-5,5)

    def sampleSO3(self, z_mu, z_std):
        '''
        Reparameterize SO(3) latent variable
        # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
        # See section 2.5 of http://ethaneade.com/lie.pdf
        '''
        # resampling trick
        if not self.training:
            return z_mu, z_std
        eps = torch.randn_like(z_std)
        w_eps = eps*z_std
        rot_eps = lie_tools.expmap(w_eps)
        #z_mu = lie_tools.quaternions_to_SO3(z_mu)
        rot_sampled = z_mu @ rot_eps
        return rot_sampled, w_eps

    def forward(self, x):
        z = self.main(x)
        z1 = z[:,:3].double()
        z2 = z[:,3:6].double()
        z_mu = lie_tools.s2s2_to_SO3(z1,z2).float()
        logvar = z[:,6:]
        z_std = torch.exp(.5*logvar) # or could do softplus
        return z_mu, z_std



