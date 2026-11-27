import pytest
import torch

from cryodrgn.config import resolve_pose_candidates
from test_spatial_radii import make_decoder


@pytest.mark.parametrize('request_mode,checkpoint,expected', [
    (None, None, 'symmetric7'),
    (None, {'epoch': 17}, 'legacy8'),
    (None, {'pose_candidates': 'symmetric7'}, 'symmetric7'),
    ('legacy8', None, 'legacy8'),
    ('symmetric7', {'epoch': 17}, 'symmetric7'),
])
def test_training_mode(request_mode, checkpoint, expected):
    assert resolve_pose_candidates(request_mode, 'grad', 'vanilla', checkpoint) == expected


def test_encoder_without_local_pose_search():
    assert resolve_pose_candidates(None, 'mlp', 'none') == 'legacy8'
    with pytest.raises(ValueError, match='symmetric7'):
        resolve_pose_candidates('symmetric7', 'mlp', 'none')


def test_symmetric_search_renders_seven_candidates(make_decoder):
    dec = make_decoder(pose_candidates='symmetric7')
    rotations = dec.symmetric_pose_rotations
    torch.testing.assert_close(rotations[0], torch.eye(3))
    torch.testing.assert_close(rotations[1:4].transpose(1, 2), rotations[4:])
    normals = rotations[1:, 2]
    tilt = torch.deg2rad(torch.tensor(3.))
    torch.testing.assert_close(normals[:, 2], tilt.cos().expand(6))
    ring = normals[:, :2] / tilt.sin()
    torch.testing.assert_close((ring * ring.roll(1, 0)).sum(1), torch.full((6,), .5))
    assert 'symmetric_pose_rotations' not in dec.state_dict()
    z = torch.randn(1, 4, requires_grad=True)
    out = dec(torch.eye(3)[None], torch.zeros(1, 2), z, euler=torch.zeros(1, 3),
              ref_fft=torch.zeros(1, 1, 16, 16), ctf=torch.ones(1, 16, 9))
    assert out['y_recon'].shape[:2] == (1, 7)
    assert torch.isfinite(torch.autograd.grad(out['y_recon'].square().mean(), z)[0]).all()
