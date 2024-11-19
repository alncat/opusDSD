'''Parse image poses from RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle

from cryodrgn import utils
from cryodrgn import lie_tools
from cryodrgn import starfile
from cryodrgn import dataset
import torch.nn.functional as F
import torch

log = utils.log

def center_of_mass(volume):
    N = volume.shape[-1]
    x_idx = torch.linspace(0, N-1, N) - N/2 #[-s, s)
    grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
    xgrid = grid[2]
    ygrid = grid[1]
    zgrid = grid[0]
    grid = torch.stack([xgrid, ygrid, zgrid], dim=-1)
    vol = ((volume > 0).float()*volume).unsqueeze(-1)
    mass = vol.sum()
    center = vol*grid
    center = center.sum(dim=(0,1,2))
    assert mass.item() > 0
    center /= mass
    #center = torch.where(center > 0, (center + 0.5).int(), (center - 0.5).int()).float()
    centered = (grid - center)
    radius = (centered).pow(2)*vol
    r = torch.sqrt(radius.sum(dim=(0,1,2))/mass)
    #principal axes
    matrix = -centered.unsqueeze(-1) * centered.unsqueeze(-2)
    radius_sum = torch.eye(3) * (radius.sum(dim=-1, keepdim=True).unsqueeze(-1))
    #print((matrix*(vol.unsqueeze(-1))).sum(dim=(0,1,2)), radius_sum.sum(dim=(0,1,2)))
    matrix = ((matrix*vol.unsqueeze(-1)+radius_sum)).sum(dim=(0, 1, 2))
    eigvals, eigvecs = np.linalg.eig(matrix.numpy())
    indices = np.argsort(eigvals)
    #print(matrix, eigvals[indices])
    eigvecs = torch.from_numpy(eigvecs[:, indices].T) # eigvecs[0] is the first eigen vector with smallest eigenvalues
    eigvals = eigvals[indices]
    #print("matrix @ eigvecs", matrix, eigvecs @ matrix @ eigvecs.T)
    new_coords = (centered)@eigvecs.T
    new_coords = -new_coords.unsqueeze(-1) * (centered @ eigvecs.T).unsqueeze(-2)
    new_coords = new_coords * vol.unsqueeze(-1)
    radius_sum = -torch.diagonal(new_coords, dim1=-2, dim2=-1).sum()
    #radius_sum = -(new_coords[..., 0, 0] + new_coords[..., 1, 1] + new_coords[..., 2, 2]).sum()
    matrix_new = new_coords.sum(dim=(0,1,2)) + torch.eye(3)*radius_sum
    matrix_new /= mass
    #print(matrix, torch.sqrt(matrix_new))
    assert np.all(eigvals >  0)
    print("r, r_p: ", r, torch.sqrt(eigvals/mass))
    r_p = torch.sqrt(eigvals/mass)

    return center, r_p, eigvecs

def add_args(parser):
    parser.add_argument('input', help='RELION .star file')
    parser.add_argument('-D', type=int, required=True, help='Box size of reconstruction (pixels)')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--Apix', type=float, help='Pixel size (A); Required if translations are specified in Angstroms')
    parser.add_argument('-o', metavar='PKL', type=os.path.abspath, required=False, help='Output pose.pkl')
    parser.add_argument('--labels', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--masks', metavar='PKL', type=os.path.abspath, required=False, help='mask starfile for multi-body refinement')
    parser.add_argument('--volumes', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--bodies', type=int, required=True, help='Number of bodies in mask starfile')
    parser.add_argument('--outmasks', default="mask_params", help="the name of pkl file storing masks related parameters")
    return parser

def main(args):
    assert args.input.endswith('.star'), "Input file must be .star file"
    #assert args.o.endswith('.pkl'), "Output format must be .pkl"

    s = starfile.Starfile.load_multibody(args.input, relion31=args.relion31)
    N = len(s.df)
    log('{} particles'.format(N))

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

    #process multibody
    log(f"there are {args.bodies} bodies")
    if s.multibodies is not None and len(s.multibodies) != 0:
        assert len(s.multibodies) == args.bodies
        body_eulers = []
        body_trans = []
        for b_i in range(args.bodies):
            body = s.multibodies[b_i]
            keys = ('_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi')
            euler_body = np.empty((N,1,3))
            assert len(body) == N
            for i in range(3):
                euler_body[:,0,i] = body[keys[i]]
            log('Euler angles (Rot, Tilt, Psi):')
            log(euler_body[0])
            body_eulers.append(euler_body)
            trans_body = np.empty((N,1,2))
            body_header = s.multibody_headers[b_i]
            if '_rlnOriginX' in body_header and '_rlnOriginY' in body_header:
                trans_body[:,0,0] = body['_rlnOriginX']
                trans_body[:,0,1] = body['_rlnOriginY']
            elif '_rlnOriginXAngst' in body_header and '_rlnOriginYAngst' in body_header:
                trans_body[:,0,0] = body['_rlnOriginXAngst']
                trans_body[:,0,1] = body['_rlnOriginYAngst']
                trans_body /= args.Apix

            log('Translations (pixels):')
            log(trans_body[0])
            trans_body /= args.D
            body_trans.append(trans_body)
    else:
        body_eulers = []
        body_trans = []
        for b_i in range(args.bodies):
            euler_body = np.zeros((N,1,3))
            euler_body[:,0,1] = 90.
            trans_body = np.zeros((N,1,2))
            body_eulers.append(euler_body)
            body_trans.append(trans_body)

    if len(body_eulers):
        body_eulers = np.concatenate(body_eulers, axis=1)
        body_trans = np.concatenate(body_trans, axis=1)
        print(body_eulers.shape, body_trans.shape)

    # write output
    if args.o is not None:
        log(f'Writing {args.o}')
        with open(args.o,'wb') as f:
            if len(body_eulers):
                pickle.dump((rot,trans,euler,body_eulers,body_trans),f)
            else:
                pickle.dump((rot,trans,euler),f)

    log(f'Loading reference volume from {args.masks}')
    s_mask = starfile.Starfile.load(args.masks)
    prefix = os.path.dirname(args.masks)
    print(s_mask.headers, prefix)
    #assert len(s_mask.df) == len(s.multibodies)
    in_relatives = []
    com_bodies = []
    radii = []
    masks = []
    axes = []
    for b_i in range(len(s_mask.df)):
        mask_name = prefix + "/" + s_mask.df['_rlnBodyMaskName'][b_i]
        in_relatives.append(int(s_mask.df['_rlnBodyRotateRelativeTo'][b_i]) - 1)
        print(mask_name)
        ref_vol = dataset.VolData(mask_name)
        masks.append(ref_vol.get())
        c, r, eigvecs = center_of_mass(ref_vol.get())#.center_of_mass()
        com_bodies.append(c)
        radii.append(r)
        axes.append(eigvecs)

    print("radii_bodies from masks: ", radii)
    masks = torch.stack(masks, dim=0)
    masks = (masks > 1e-3)*masks
    vol_coms = None
    rot_radii = None
    if args.volumes:
        #read in dynamics volumes
        vols = []
        for b_i in range(10):
            mask_name = args.volumes + "/reference" + str(b_i) + ".mrc"
            print(mask_name)
            ref_vol = dataset.VolData(mask_name)
            vols.append(ref_vol.get())
            if b_i == 0:
                #interpolate mask
                #scale = masks.shape[-1]/vols[-1].shape[-1]
                #assert args.D == masks.shape[-1]
                #scale = args.D/vols[-1].shape[-1]
                #scale_apix = ref_vol.Apix/args.Apix
                # TODO: the volumes are cropped, but masks of full size are not cropped
                print(f"need to resample masks from {args.Apix} to {ref_vol.Apix}")
                print(f"mask length {args.D*args.Apix}, volume length {vols[-1].shape[-1]*ref_vol.Apix}")
                crop_size = int((args.D*args.Apix - vols[-1].shape[-1]*ref_vol.Apix)/args.Apix)//2
                if masks.shape[-1] == args.D:
                    print(f"need to crop masks by {crop_size}")
                    masks = masks[:, crop_size:args.D-crop_size, crop_size:args.D-crop_size, crop_size:args.D-crop_size]
                    assert masks.shape[-1] == args.D - crop_size*2
                print(f"mask shape after cropping {masks.shape}")
                #scale = masks.shape[-1]/vols[-1].shape[-1]
                scale = (args.D - crop_size*2)/vols[-1].shape[-1]
                print(f"rescale the coordinates by {scale}")
                masks = F.interpolate(masks.unsqueeze(0), vols[-1].shape, mode='trilinear').squeeze()
                print(masks.sum(dim=(1,2,3)))

        c0s = []
        c1s = []
        vol_coms = []
        #reset radii
        radii = []
        principal_axes = []
        for m_i in range(masks.shape[0]):
            c0, r0, p0 = center_of_mass(vols[0]*masks[m_i])
            c1, r1, p1 = center_of_mass(vols[-1]*masks[m_i])
            c0 *= scale
            c1 *= scale
            print(r0*scale, r1*scale)
            #print(p1@p0.T)
            c0s.append(c0)
            c1s.append(c1)
            #print(c0, c1)
            vol_com, r, p_axes = center_of_mass(vols[4]*masks[m_i])
            radii.append(r*scale)
            vol_coms.append(vol_com*scale)
            principal_axes.append(p_axes)

        rot_axes = []
        orientations = []
        rot_radii = []
        origin_rel = np.bincount(in_relatives).argmax()
        for m_i in range(masks.shape[0]):
            r0 = com_bodies[in_relatives[m_i]] - c0s[m_i]
            r1 = com_bodies[in_relatives[m_i]] - c1s[m_i]
            rot_axis = torch.cross(r0, r1, dim=-1)
            rot_axis = F.normalize(rot_axis, dim=0)
            r0 = F.normalize(r0, dim=0)
            rot_axes.append(rot_axis)
            r1 = torch.cross(r0, rot_axis, dim=-1)
            r1 = F.normalize(r1, dim=0)
            mat = torch.stack([r0, r1, rot_axis], dim=0)
            orientations.append(mat)
            print(mat@principal_axes[m_i].T)
            if m_i == origin_rel:
                rot_radii.append(vol_coms[m_i] - vol_coms[m_i])
            else:
                rot_radii.append(vol_coms[m_i] - vol_coms[in_relatives[m_i]])
            #print(rot_axis, mat)
            #print(mat@rot_axis)
            #print(mat@mat.T)

        rot_axes = torch.stack(rot_axes, dim=0)
        orientations = torch.stack(orientations, dim=0)
        vols = torch.stack(vols, dim=0)
        rot_radii = torch.stack(rot_radii, dim=0)
        vol_coms = torch.stack(vol_coms, dim=0)
        principal_axes = torch.stack(principal_axes, dim=0)

    consensus_mask = masks.mean(dim=0)
    #weights = F.softmax(masks*4, dim=0)
    #print(weights.shape)

    com_bodies = torch.stack(com_bodies, dim=0)
    if vol_coms is None:
        vol_coms = com_bodies
    radii_bodies = torch.stack(radii, dim=0)
    rotate_directions = []
    rotate_directions_ori = []
    orient_bodies = []
    relats = []
    print("in_relatives: ", in_relatives)
    print("coms computed from masks: ", com_bodies,)
    print("coms computed from volumes: ", vol_coms)
    origin_rel = np.bincount(in_relatives).argmax()
    for b_i in range(len(s_mask.df)):
        rotate_directions.append(com_bodies[in_relatives[b_i]] - com_bodies[b_i])
        rotate_directions_ori.append(com_bodies[b_i] - com_bodies[in_relatives[b_i]])
        rotate_directions[-1] = F.normalize(rotate_directions[-1], dim=0)
        if b_i != origin_rel:
            orient_bodies.append(utils.align_with_z(-rotate_directions[-1]))
        else:
            orient_bodies.append(utils.align_with_z(rotate_directions[-1]))
        relats.append(com_bodies[in_relatives[b_i]])
        #reset rotation axis for center mask
        #if b_i == origin_rel:
        #    rotate_directions_ori[b_i] = com_bodies[b_i] - com_bodies[b_i]
        #normalize direction
    A_rot90 = lie_tools.yrot(torch.tensor(-90))
    rotate_directions = torch.stack(rotate_directions, dim=0)
    rotate_directions_ori = torch.stack(rotate_directions_ori, dim=0)
    if rot_radii is None:
        rot_radii = rotate_directions
    #print((orientations@rotate_directions_ori.unsqueeze(-1)).squeeze(), rot_axes, orientations)
    #print((orientations@rot_radii.unsqueeze(-1)).squeeze())
    #print(orientations@torch.transpose(principal_axes, -1, -2))
    orient_bodies = torch.stack(orient_bodies, dim=0)
    relats = torch.stack(relats, dim=0)
    axes = torch.stack(axes, dim=0)
    #print("A_rot90: ", A_rot90)
    #print("relats: ", relats)
    print("rotate_directions determined by masks: ", rotate_directions_ori)
    print("orient_bodies for translation by aligning difference of coms to z axis: ", orient_bodies)
    print("principal_axes from masks: ", axes)
    #print("radii_bodies from masks: ", radii_bodies)
    output_name = prefix + f"/{args.outmasks}.pkl"
    log(f'Writing parameters to {output_name}')
    if not args.volumes:
        torch.save({"in_relatives": relats, "com_bodies": com_bodies,
                "orient_bodies": orient_bodies, "rotate_directions": rotate_directions_ori, "radii_bodies": radii_bodies, "principal_axes": axes}, \
    #            #"weights": weights, "consensus_mask": consensus_mask},
               output_name)
    else:
        print("rotate_directions using volumes: ", rot_radii)
        print("orient_bodies for translation determined from volume series: ", orientations)
        print("principal_axes from volumes: ", principal_axes)
        print("radii_bodies from volumes: ", radii_bodies)
        torch.save({"in_relatives": relats, "com_bodies": vol_coms,
                "orient_bodies": orientations, "rotate_directions": rot_radii, "radii_bodies": radii_bodies, "principal_axes": principal_axes,},  \
                #"weights": weights, "consensus_mask": consensus_mask},
               output_name)
    # shift of each rigid body in experimental data is selfRound(my_old_offset - Aori*com) + ibody_offset
    # backproject into reference model should be, Aresi(x-com) + com - Inv(Aori)*ibody_offset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
