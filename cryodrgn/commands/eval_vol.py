'''
Evaluate the decoder at specified values of z
'''
import numpy as np
import sys, os
import argparse
import pickle
from datetime import datetime as dt
import pprint

import torch
import torch.nn as nn

from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import fft
from cryodrgn import lie_tools
from cryodrgn import config
from cryodrgn.lattice import Lattice
from cryodrgn.models import HetOnlyVAE

log = utils.log
vlog = utils.vlog

def add_args(parser):
    #parser.add_argument('weights', help='Model weights')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('-c', '--config', metavar='PKL', required=True, help='CryoDRGN config.pkl file')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc or directory')
    parser.add_argument('--prefix', default='vol_', help='Prefix when writing out multiple .mrc files (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')

    group = parser.add_argument_group('Specify z values')
    group.add_argument('-z', type=np.float32, nargs='*', help='Specify one z-value')
    group.add_argument('--z-start', type=np.float32, nargs='*', help='Specify a starting z-value')
    group.add_argument('--z-end', type=np.float32, nargs='*', help='Specify an ending z-value')
    group.add_argument('-n', type=int, default=10, help='Number of structures between [z_start, z_end]')
    group.add_argument('--zfile', help='Text file with z-values to evaluate')

    group = parser.add_argument_group('Volume arguments')
    group.add_argument('--Apix', type=float, default=1, help='Pixel size to add to .mrc header (default: %(default)s A/pix)')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('-d','--downsample', type=int, help='Downsample volumes to this box size (pixels)')

    group = parser.add_argument_group('Overwrite architecture hyperparameters in config.pkl')
    group.add_argument('--norm', nargs=2, type=float)
    group.add_argument('-D', type=int, help='Box size')
    group.add_argument('--enc-layers', dest='qlayers', type=int, help='Number of hidden layers')
    group.add_argument('--enc-dim', dest='qdim', type=int, help='Number of nodes in hidden layers')
    group.add_argument('--zdim', type=int,  help='Dimension of latent variable')
    group.add_argument('--encode-mode', choices=('conv','resid','mlp','tilt', 'grad'), help='Type of encoder network')
    group.add_argument('--dec-layers', dest='players', type=int, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, help='Number of nodes in hidden layers')
    group.add_argument('--enc-mask', type=int, help='Circular mask radius for image encoder')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf','none', 'vanilla'), help='Type of positional encoding')
    group.add_argument('--template-type', choices=('conv'), help='Type of template decoding method (default: %(default)s)')
    group.add_argument('--warp-type', choices=('blurmix', 'diffeo', 'deform'), help='Type of warp decoding method (default: %(default)s)')
    group.add_argument('--symm', help='Type of symmetry of the 3D volume (default: %(default)s)')
    group.add_argument('--num-struct', type=int, default=1, help='Num of structures (default: %(default)s)')
    group.add_argument('--deform-size', type=int, default=2, help='Num of structures (default: %(default)s)')

    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--domain', choices=('hartley','fourier'))
    group.add_argument('--l-extent', type=float, help='Coordinate lattice size')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    return parser

def check_inputs(args):
    if args.z_start:
        assert args.z_end, "Must provide --z-end with argument --z-start"
    assert sum((bool(args.z), bool(args.z_start), bool(args.zfile))) == 1, "Must specify either -z OR --z-start/--z-end OR --zfile"

def main(args):
    #check_inputs(args)
    t1 = dt.now()

    ## set the device
    use_cuda = torch.cuda.is_available()
    log('Use cuda {}'.format(use_cuda))
    device = torch.device('cuda' if use_cuda else 'cpu')
    #if use_cuda:
    #    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #else:
    #    log('WARNING: No GPUs detected')

    log(args)
    cfg = config.overwrite_config(args.config, args)
    log('Loaded configuration:')
    pprint.pprint(cfg)

    in_dim = -1
    enc_mask = -1
    D = cfg['lattice_args']['D'] # image size + 1
    zdim = cfg['model_args']['zdim']
    norm = cfg['dataset_args']['norm']
    lattice = Lattice(D, extent=0.5)
    downfrac = cfg['dataset_args']['downfrac']
    down_vol_size = cfg['model_args']['down_vol_size']
    Apix = cfg['model_args']['Apix']
    templateres = cfg['model_args']['templateres']
    #args.Apix = down_vol_size/((D - 1)*downfrac*0.85)*Apix
    window_r = down_vol_size/((D-1)*downfrac)
    downfrac *= Apix/args.Apix
    log("Apix: changing from training apix {} to target apix {}".format(Apix, args.Apix))
    log("the output volume by convnet will further downsample by downfrac: {} to achieve desired apix".format(downfrac))
    assert templateres is not None
    log("templateres: output volume of convnet is of size {}".format(templateres))
    log("the final output volume rendered by spatial transformer is of size {}".format(int((D-1)*downfrac*window_r)))

    if args.downsample:
        assert args.downsample % 2 == 0, "Boxsize must be even"
        assert args.downsample <= D - 1, "Must be smaller than original box size"
    #create and load model
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                in_dim, zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                activation=activation, ref_vol=None, Apix=args.Apix,
                template_type=args.template_type, warp_type=args.warp_type,
                num_struct=args.num_struct,
                device=device, symm=args.symm, ctf_grid=None,
                deform_emb_size=args.deform_size, downfrac=downfrac,
                templateres=templateres, window_r=window_r)

    vanilla = args.pe_type == "vanilla"

    if args.load:
        log('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        print(checkpoint.keys())
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        #print(pretrained_dict, model_dict)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        if vanilla:
            pretrained_dict = checkpoint['encoder_state_dict']
            model_dict = model.encoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "grid" not in k and "mask" not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            #model.encoder.load_state_dict(model_dict)

            pretrained_dict = checkpoint['decoder_state_dict']
            model_dict = model.decoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "grid" not in k and "mask" not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.decoder.load_state_dict(model_dict)

    model = model.to(device)

    model.eval()

    ### Multiple z ###
    if args.z_start or args.zfile:

        ### Get z values
        if args.z_start:
            args.z_start = np.array(args.z_start)
            args.z_end = np.array(args.z_end)
            z = np.repeat(np.arange(args.n,dtype=np.float32), zdim).reshape((args.n, zdim))
            z *= ((args.z_end - args.z_start)/(args.n-1))
            z += args.z_start
        else:
            if vanilla:
                #z = utils.load_pkl(args.zfile)
                z = np.loadtxt(args.zfile).reshape(-1, zdim)
                z = torch.tensor(z).float().to(device)
            else:
                z = np.loadtxt(args.zfile).reshape(-1, zdim)

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        log(f'Generating {len(z)} volumes')
        for i,zz in enumerate(z):
            log(zz)
            if vanilla:
                model.save_mrc(f'{args.o}/reference'+str(i), enc=zz, Apix=args.Apix)
            else:
                if args.downsample:
                    extent = lattice.extent * (args.downsample/(D-1))
                    vol = model.decoder.eval_volume(lattice.get_downsample_coords(args.downsample+1),
                                                    args.downsample+1, extent, norm, zz)
                else:
                    vol = model.decoder.eval_volume(lattice.coords, lattice.D, lattice.extent, norm, zz)
                out_mrc = '{}/{}{:03d}.mrc'.format(args.o, args.prefix, i)
                if args.flip:
                    vol = vol[::-1]
                mrc.write(out_mrc, vol.astype(np.float32), Apix=args.Apix)

    ### Single z ###
    else:
        #z = np.array(args.z)
        z = torch.randn(1, args.zdim).to(device)
        log(z)
        if vanilla:
            model.save_mrc('reference', enc=z)
            return
        if args.downsample:
            extent = lattice.extent * (args.downsample/(D-1))
            vol = model.decoder.eval_volume(lattice.get_downsample_coords(args.downsample+1),
                                            args.downsample+1, extent, norm, z)
        else:
            vol = model.decoder.eval_volume(lattice.coords, lattice.D, lattice.extent, norm, z)
        if args.flip:
            vol = vol[::-1]
        mrc.write(args.o, vol.astype(np.float32), Apix=args.Apix)

    td = dt.now()-t1
    log('Finished in {}'.format(td))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)

