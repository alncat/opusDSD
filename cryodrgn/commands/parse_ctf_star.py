'''Parse CTF parameters from a RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle
import bisect

from cryodrgn import utils
from cryodrgn import starfile
from cryodrgn import ctf
log = utils.log

HEADERS = ['_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift']

def add_args(parser):
    parser.add_argument('star', help='Input')
    parser.add_argument('--Apix', type=float, required=True, help='Angstroms per pixel')
    parser.add_argument('-D', type=int, required=True, help='Image size in pixels')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output pkl of CTF parameters')
    parser.add_argument('-o-g', type=os.path.abspath, required=True, help='Output pkl of group assignments')
    parser.add_argument('--png', metavar='PNG', type=os.path.abspath, help='Optionally plot the CTF')

    group = parser.add_argument_group('Overwrite CTF parameters')
    group.add_argument('--kv', type=float, help='Accelerating voltage (kV)')
    group.add_argument('--cs', type=float, help='Spherical abberation (mm)')
    group.add_argument('-w', type=float, help='Amplitude contrast ratio')
    group.add_argument('--ps', type=float, help='Phase shift (deg)')
    return parser

def main(args):
    assert args.star.endswith('.star'), "Input file must be .star file"
    assert args.o.endswith('.pkl'), "Output CTF parameters must be .pkl file"
    assert args.o_g.endswith('.pkl'), "Output group assignment must be .pkl file"
    if args.relion31: # TODO: parse the data_optics block
        assert args.kv is not None, "--kv must be set manually with RELION 3.1 file format"
        assert args.cs is not None, "--cs must be set manually with RELION 3.1 file format"
        assert args.w is not None, "-w must be set manually with RELION 3.1 file format"

    s = starfile.Starfile.load(args.star, relion31=args.relion31)
    N = len(s.df)
    log('{} particles'.format(N))

    overrides = {}
    if args.kv is not None:
        log(f'Overriding accerlating voltage with {args.kv} kV')
        overrides[HEADERS[3]] = args.kv
    if args.cs is not None:
        log(f'Overriding spherical abberation with {args.cs} mm')
        overrides[HEADERS[4]] = args.cs
    if args.w is not None:
        log(f'Overriding amplitude contrast ratio with {args.w}')
        overrides[HEADERS[5]] = args.w
    if args.ps is not None:
        log(f'Overriding phase shift with {args.ps}')
        overrides[HEADERS[6]] = args.ps

    ctf_params = np.zeros((N, 9))

    ctf_params[:,0] = args.D
    ctf_params[:,1] = args.Apix
    for i,header in enumerate(['_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift']):
        ctf_params[:,i+2] = s.df[header] if header  not in overrides else overrides[header]

    particle_groups = np.random.randint(N, size=(N)) #if "_rlnGroupNumber" in s.df s.df['_rlnGroupNumber'].astype(np.int64)
    particle_groups_from_defocus = np.zeros((N), dtype=np.int32)
    all_defocus = []
    group_count = 1
    group_set   = {}
    mapping = {}
    part_set = set()
    part_set.add(1)

    for i in range(N):
        avg_def = ctf_params[i, 2] + ctf_params[i, 3]
        avg_def /= 2.
        #if particle_groups[i] not in part_set:
        #    if (max(part_set) != particle_groups[i] - 1):
        #        print("record order: ", particle_groups[i], max(part_set))
        #    part_set.add(particle_groups[i])

        if i == 0:
            all_defocus.append(avg_def)
            group_set[avg_def] = group_count
            particle_groups_from_defocus[i] = group_set[avg_def]
        else:
            i_le = bisect.bisect_left(all_defocus, avg_def)
            if i_le == len(all_defocus):
                i_le -= 1

            if abs(all_defocus[i_le] - avg_def) <= 10:
                particle_groups_from_defocus[i] = group_set[all_defocus[i_le]]
            elif i_le and abs(all_defocus[i_le - 1] - avg_def) <= 10:
                particle_groups_from_defocus[i] = group_set[all_defocus[i_le - 1]]
            else:
                group_count += 1
                group_set[avg_def] = group_count
                particle_groups_from_defocus[i] = group_set[avg_def]
                bisect.insort_left(all_defocus, avg_def)

            if particle_groups_from_defocus[i] != particle_groups[i]:
                if particle_groups[i] not in part_set:
                    part_set.add(particle_groups[i])
                if particle_groups[i] not in mapping:
                    mapping[particle_groups[i]] = particle_groups_from_defocus[i]
                    #not newly added
                    #if particle_groups_from_defocus[i] != group_count:
                    #    print(particle_groups[i], group_count, avg_def, i_le, all_defocus[i_le], all_defocus[i_le-1])
                    #print(particle_groups_from_defocus[i], particle_groups[i], particle_groups_from_defocus[i] - particle_groups[i])
                # when mapping doesn't match, then they do not match
                if mapping[particle_groups[i]] != particle_groups_from_defocus[i]:
                    print("duplicate: ", particle_groups_from_defocus[i], particle_groups[i], particle_groups_from_defocus[i] - particle_groups[i])
    print(set(particle_groups))
    print(mapping)

    log('CTF parameters for first particle:')
    ctf.print_ctf_params(ctf_params[0])
    log('Saving {}'.format(args.o))
    with open(args.o,'wb') as f:
        pickle.dump(ctf_params.astype(np.float32), f)
    log('Saving {}'.format(args.o_g))
    with open(args.o_g, 'wb') as f:
        pickle.dump(particle_groups.astype(np.int64), f)
    if args.png:
        import matplotlib.pyplot as plt
        assert args.D, 'Need image size to plot CTF'
        ctf.plot_ctf(args.D, args.Apix, ctf_params[0,2:])
        plt.savefig(args.png)
        log(args.png)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
