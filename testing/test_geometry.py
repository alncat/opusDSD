'''
The box size and the pixel size of the volumes written by eval_vol

A model renders the volume of interest which was cropped out of the downsampled images during
training, so the extent it covers in angstrom is fixed. Rendering at another pixel size means
resampling that same extent onto another number of voxels. These tests build the model the way
the trainer does, write the config the trainer writes, and check that eval_vol derives a box and
a pixel size which still describe the same physical object. They run on the cpu in a few seconds.
'''
import contextlib
import io

import pytest
import torch.nn as nn

from cryodrgn.commands import eval_vol
from cryodrgn.lattice import Lattice
from cryodrgn.models import HetOnlyVAE

# box, downfrac, window_r, apix of the particle stack
SETTINGS = [
    (256, 0.5, 0.85, 1.0),    # (box*downfrac) is even, the common case
    (250, 0.5, 0.85, 1.0),    # (box*downfrac) is an odd integer
    (256, 0.53, 0.85, 1.06),  # (box*downfrac) is not an integer at all
    (320, 0.75, 0.725, 1.699),# the spliceosome tutorial dataset
    (128, 0.9, 0.85, 3.0),    # small box, little downsampling
]
TEMPLATERES = 64  # only sets the size of the rendered template, keeps the test cheap


def build_model(box, downfrac, window_r, apix):
    '''Instantiate the model the way both trainers and eval_vol do'''
    with contextlib.redirect_stdout(io.StringIO()):  # the model logs a lot while building
        return HetOnlyVAE(Lattice(box + 1, extent=0.5), 3, 256, 3, 256, -1, 8,
                          encode_mode='grad', enc_mask=-1, enc_type='vanilla', enc_dim=None,
                          domain='fourier', activation=nn.ReLU, ref_vol=None, Apix=apix,
                          template_type='conv', warp_type=None, num_struct=1, device='cpu',
                          symm=None, ctf_grid=None, deform_emb_size=2, downfrac=downfrac,
                          templateres=TEMPLATERES, window_r=window_r, masks_params=None,
                          num_bodies=0, z_affine_dim=4)


def train(box, downfrac, window_r, apix):
    '''Return the trained model together with the config which save_config would write for it'''
    model = build_model(box, downfrac, window_r, apix)
    cfg = dict(lattice_args=dict(D=box + 1, extent=0.5, ignore_DC=True),
               dataset_args=dict(downfrac=downfrac, norm=[0, 1.0], window_r=0.85),
               model_args=dict(zdim=8, z_affine_dim=4, down_vol_size=model.down_vol_size,
                               Apix=float(model.decoder.Apix), templateres=TEMPLATERES))
    return model, cfg


def render_at(cfg, target_apix):
    '''Return the box and the pixel size eval_vol would write for this target pixel size'''
    geometry = eval_vol.output_geometry(cfg, target_apix)
    model = build_model(cfg['lattice_args']['D'] - 1, geometry['downfrac'],
                        geometry['window_r'], target_apix)
    return model.down_vol_size, geometry['extent'] / model.down_vol_size


@pytest.mark.parametrize('settings', SETTINGS, ids=lambda s: 'box{}_downfrac{}'.format(s[0], s[1]))
def test_training_pixel_size_reproduces_the_training_box(settings):
    '''Asking for the pixel size the model was trained at has to give the box it was trained at

    Recovering the cropping fraction from the config used to divide by (D-1)*downfrac, which is
    not what the model divided by when it was built, so the box came out two pixels short.
    '''
    model, cfg = train(*settings)
    box, apix = render_at(cfg, cfg['model_args']['Apix'])
    assert box == model.down_vol_size
    assert apix == pytest.approx(cfg['model_args']['Apix'])


@pytest.mark.parametrize('settings', SETTINGS, ids=lambda s: 'box{}_downfrac{}'.format(s[0], s[1]))
@pytest.mark.parametrize('scale', [1.0, 0.5, 0.971])  # 0.971 lands the box between two even sizes
def test_written_pixel_size_describes_the_true_scale(settings, scale):
    '''box * written apix has to stay the extent which was cropped out during training

    The box is rounded down to an even number, so the pixel size which can actually be realized
    is not exactly the requested one. Writing the requested one into the header instead of the
    realized one silently rescales the map by up to a percent.
    '''
    _, cfg = train(*settings)
    target = cfg['model_args']['Apix'] * scale
    box, apix = render_at(cfg, target)
    extent = cfg['model_args']['down_vol_size'] * cfg['model_args']['Apix']
    assert box * apix == pytest.approx(extent)
    # the requested pixel size can be missed, the render size and the box are each rounded down
    # to an even number and the two losses add up, but never by more than those few voxels
    assert abs(apix - target) / target < 4.0 / box


@pytest.mark.parametrize('settings', SETTINGS, ids=lambda s: 'box{}_downfrac{}'.format(s[0], s[1]))
def test_box_is_even(settings):
    '''An odd box would not survive the centered ffts downstream'''
    _, cfg = train(*settings)
    for scale in (1.0, 0.6):
        box, _ = render_at(cfg, cfg['model_args']['Apix'] * scale)
        assert box % 2 == 0


def test_a_smaller_pixel_size_gives_a_bigger_box():
    _, cfg = train(*SETTINGS[0])
    apix = cfg['model_args']['Apix']
    boxes = [render_at(cfg, apix * scale)[0] for scale in (1.0, 0.5, 0.25)]
    assert boxes[0] < boxes[1] < boxes[2]


def test_output_geometry_reads_a_tensor_apix():
    '''Configs written before Apix was stored as a float keep it as a tensor on the training gpu'''
    import torch
    _, cfg = train(*SETTINGS[0])
    apix = cfg['model_args']['Apix']
    cfg['model_args']['Apix'] = torch.tensor(apix)
    geometry = eval_vol.output_geometry(cfg, apix)
    assert isinstance(geometry['extent'], float)
    assert geometry['train_apix'] == pytest.approx(apix)
