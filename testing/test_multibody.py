'''
The rigid body parameters parse_multi_pose_star measures off the masks

The principal axes it returns are used as a rotation by the decoder, so they have to be an
orthonormal right handed frame for every body shape, including the globular ones whose moments of
inertia are nearly degenerate.
'''
import contextlib
import io

import numpy as np
import pytest
import torch

from cryodrgn.commands.parse_multi_pose_star import center_of_mass, pick_origin_body

N = 32  # box size, keeps the 3d grids small


def ellipsoid(semi_axes, center=(0., 0., 0.), rotation=None):
    '''A solid ellipsoid with the given semi axes, optionally rotated and off center'''
    idx = torch.linspace(0, N - 1, N) - N / 2
    zgrid, ygrid, xgrid = torch.meshgrid(idx, idx, idx, indexing='ij')
    coords = torch.stack([xgrid, ygrid, zgrid], dim=-1) - torch.tensor(center)
    if rotation is not None:
        coords = coords @ rotation  # rotate the coordinates, i.e. rotate the body the other way
    scaled = coords / torch.tensor(semi_axes)
    return (scaled.pow(2).sum(-1) < 1).float()


def rotation_from_quaternion(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
                        dtype=torch.float32)


def com(volume):
    with contextlib.redirect_stdout(io.StringIO()):  # it prints the radii it measured
        return center_of_mass(volume)


def test_center_of_mass_finds_the_center():
    center, _, _ = com(ellipsoid((10., 6., 4.), center=(3., -5., 2.)))
    assert center.numpy() == pytest.approx([3., -5., 2.], abs=0.2)


def test_radii_are_ordered_and_reflect_the_shape():
    '''A rod has a small radius of gyration about its long axis and a large one about the others'''
    _, radii, _ = com(ellipsoid((14., 3., 3.)))
    assert radii[0] < radii[1]
    assert radii[1] == pytest.approx(radii[2], rel=0.05)


def test_axes_are_orthonormal_and_right_handed():
    for semi_axes in [(10., 6., 4.),      # three distinct moments
                      (12., 12., 3.),     # oblate, two moments equal
                      (14., 3., 3.),      # prolate, two moments equal
                      (8., 8., 8.)]:      # a sphere, all three degenerate
        _, _, axes = com(ellipsoid(semi_axes))
        assert torch.det(axes) == pytest.approx(1.0, abs=1e-4), \
            'axes of {} are a reflection'.format(semi_axes)
        assert (axes @ axes.T - torch.eye(3)).abs().max() < 1e-4, \
            'axes of {} are not orthonormal'.format(semi_axes)


@pytest.mark.parametrize('seed', range(8))
def test_axes_stay_right_handed_under_rotation(seed):
    '''np.linalg.eig used to return a reflection for about half of the orientations'''
    rotation = rotation_from_quaternion(np.random.default_rng(seed).normal(size=4))
    _, _, axes = com(ellipsoid((11., 7., 5.), rotation=rotation))
    assert torch.det(axes) == pytest.approx(1.0, abs=1e-4)
    assert (axes @ axes.T - torch.eye(3)).abs().max() < 1e-4


def multibody_model(with_masks, num_bodies=4, zdim=12, z_affine_dim=6):
    from cryodrgn.lattice import Lattice
    from cryodrgn.models import HetOnlyVAE
    import torch.nn as nn
    masks = None
    if with_masks:
        masks = dict(com_bodies=torch.randn(num_bodies, 3) * 10,
                     in_relatives=torch.randn(num_bodies, 3),
                     rotate_directions=torch.randn(num_bodies, 3),
                     orient_bodies=torch.eye(3).repeat(num_bodies, 1, 1),
                     principal_axes=torch.eye(3).repeat(num_bodies, 1, 1),
                     radii_bodies=torch.rand(num_bodies, 3) * 20 + 5)
    with contextlib.redirect_stdout(io.StringIO()):
        model = HetOnlyVAE(Lattice(129, extent=0.5), 3, 256, 3, 256, -1, zdim, encode_mode='grad',
                           enc_mask=-1, enc_type='vanilla', enc_dim=None, domain='fourier',
                           activation=nn.ReLU, ref_vol=None, Apix=2.0, template_type='conv',
                           warp_type=None, num_struct=1, device='cpu', symm=None, ctf_grid=None,
                           deform_emb_size=2, downfrac=0.5, templateres=64, window_r=0.85,
                           masks_params=masks, num_bodies=num_bodies, z_affine_dim=z_affine_dim)
    model.eval()
    return model


@pytest.mark.parametrize('z_width', [12, 18], ids=['latent_without_body_motion', 'latent_with_body_motion'])
def test_decoder_writes_a_volume_on_the_cpu(z_width, tmp_path):
    '''The decoder used to ask a cpu tensor for its device index, which is -1

    Rendering volumes is a forward pass of the template network, there is no reason it should
    need a gpu, and the analysis of a finished run is often done on a laptop.
    '''
    from cryodrgn import mrc
    model = multibody_model(with_masks=True)
    with torch.no_grad():
        model.decoder.save(str(tmp_path / 'v'), z=torch.randn(1, z_width), Apix=2.0)
    volume, header = mrc.parse_mrc(str(tmp_path / 'v.mrc'))
    assert volume.shape == (54, 54, 54)
    assert header.get_apix() == pytest.approx(2.0)


def test_body_motion_without_the_body_geometry_says_what_is_missing(tmp_path):
    '''--num-bodies alone builds the head but not the geometry the bodies move with'''
    model = multibody_model(with_masks=False)
    with pytest.raises(RuntimeError, match='--masks'):
        with torch.no_grad():
            model.decoder.save(str(tmp_path / 'v'), z=torch.randn(1, 18), Apix=2.0)


def test_origin_body_is_the_most_referenced_one():
    '''_rlnBodyRotateRelativeTo of four bodies which all hang off the second one'''
    assert pick_origin_body([1, 1, 1, 1]) == 1


def test_origin_body_can_be_given(capsys):
    assert pick_origin_body([1, 1, 1, 1], requested=3) == 2  # the starfile counts from one
    assert 'body 3 is the origin' in capsys.readouterr().out


def test_a_tie_between_bodies_is_reported(capsys):
    '''Two bodies referenced twice each, argmax settles on the first without saying so'''
    origin = pick_origin_body([1, 1, 3, 3])
    assert origin == 1
    warning = capsys.readouterr().out
    assert 'WARNING' in warning
    assert '[2, 4]' in warning
    assert '--origin-body' in warning


def test_origin_body_out_of_range_is_refused():
    for requested in (0, 5):
        with pytest.raises(AssertionError, match='origin-body'):
            pick_origin_body([1, 1, 1, 1], requested=requested)


def test_axes_recover_the_orientation_of_the_body():
    '''The longest axis of the body has the smallest moment, so it comes first'''
    rotation = rotation_from_quaternion([1., 0.3, -0.2, 0.5])
    _, _, axes = com(ellipsoid((13., 6., 4.), rotation=rotation))
    # coords @ rotation applies the transpose to the coordinates, so the body itself is turned by
    # rotation and its long axis, x before the rotation, ends up along the first column
    long_axis = rotation @ torch.tensor([1., 0., 0.])
    assert abs(torch.dot(axes[0].float(), long_axis)) == pytest.approx(1.0, abs=0.02)
