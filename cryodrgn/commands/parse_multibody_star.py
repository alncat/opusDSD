'''Parse image poses from RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle
import dataset
import torch.nn.functional as F

from cryodrgn import utils
from cryodrgn import starfile
log = utils.log

def add_args(parser):
    parser.add_argument('input', help='RELION .star file')
    parser.add_argument('-D', type=int, required=True, help='Box size of reconstruction (pixels)')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--Apix', type=float, help='Pixel size (A); Required if translations are specified in Angstroms')
    parser.add_argument('-o', metavar='PKL', type=os.path.abspath, required=False, help='Output pose.pkl')
    parser.add_argument('--labels', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--outdir', type=os.path.abspath)
    return parser

def compute_rot_dirs(masks, relatives_to):
    coms = []
    for mask in masks:
        coms.append(mask.center_of_mass())
    rot_dirs = []
    orients = []
    for i in range(len(relatives_to)):
        if relatives_to[i] > 0:
            rot_dirs.append(coms[relatives_to[i]] - coms[i])
        else:
            rot_dirs.append(-coms[i])
    for i in range(len(rot_dirs)):
        rot_dirs[i] = F.normalize(rot_dirs[i], dim=0)
    for i in range(len(rot_dirs)):
        orients.append(utils.align_with_z(rot_dirs[i]))
    return coms

def main(args):
    assert args.input.endswith('.star'), "Input file must be .star file"
    #assert args.o.endswith('.pkl'), "Output format must be .pkl"

    s = starfile.Starfile.load(args.input, relion31=args.relion31)
    N = len(s.df)
    log('{} particles'.format(N))
    # read in all masks and calculate com

    masks = []
    masks = dataset.VolData(args.mask_vol).get()

    # parse rotations
    keys = ('_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi')
    euler = np.empty((N,3))
    euler[:,0] = s.df['_rlnAngleRot']
    euler[:,1] = s.df['_rlnAngleTilt']
    euler[:,2] = s.df['_rlnAnglePsi']
    log('Euler angles (Rot, Tilt, Psi):')
    log(euler[0])
    log('Converting to rotation matrix:')
    rot = np.asarray([utils.R_from_relion(*x) for x in euler])
    log(rot[0])
    if args.labels is not None:
        labels = utils.load_pkl(args.labels)
        log(f'Read labels from {args.labels}')
        for i in range(labels.min(), labels.max()+1):
            out_file = args.outdir + "/pre" + str(i) + ".star"
            log(f'Writing {np.sum(labels==i)} particles in cluster {i} to {out_file}')
            s.write_subset(out_file, labels==i)

    # parse translations
    trans = np.empty((N,2))
    if '_rlnOriginX' in s.headers and '_rlnOriginY' in s.headers:
        trans[:,0] = s.df['_rlnOriginX']
        trans[:,1] = s.df['_rlnOriginY']
    elif '_rlnOriginXAngst' in s.headers and '_rlnOriginYAngst' in s.headers:
        assert args.Apix is not None, "Must provide --Apix argument to convert _rlnOriginXAngst and _rlnOriginYAngst translation units"
        trans[:,0] = s.df['_rlnOriginXAngst']
        trans[:,1] = s.df['_rlnOriginYAngst']
        trans /= args.Apix

    log('Translations (pixels):')
    log(trans[0])

    # convert translations from pixels to fraction
    trans /= args.D

    # write output
    if args.o is not None:
        log(f'Writing {args.o}')
        with open(args.o,'wb') as f:
            pickle.dump((rot,trans,euler),f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
