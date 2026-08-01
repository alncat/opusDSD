'''
Loading weights out of a checkpoint, and reading configs written on another machine

A model built with different architecture arguments than the ones it was trained with used to
load whatever happened to fit and leave the rest at its initial value, which produces volumes
that look plausible but are missing whatever the unloaded weights represented.
'''
import contextlib
import io
import pickle

import pytest
import torch
import torch.nn as nn

from cryodrgn import utils
from cryodrgn.lattice import Lattice
from cryodrgn.models import HetOnlyVAE


class Small(nn.Module):
    def __init__(self, out=4, buffer_size=6):
        super().__init__()
        self.lin = nn.Linear(4, out)
        self.register_buffer('grid', torch.zeros(buffer_size))


def test_loads_everything_from_a_matching_checkpoint():
    src, dst = Small(), Small()
    n_params, n_buffers = utils.load_matching_state_dict(dst, src.state_dict(), name='small')
    assert (n_params, n_buffers) == (2, 1)
    assert torch.equal(dst.lin.weight, src.lin.weight)


def test_a_parameter_which_does_not_fit_is_an_error():
    src, dst = Small(out=4), Small(out=8)
    with pytest.raises(RuntimeError) as e:
        utils.load_matching_state_dict(dst, src.state_dict(), name='small')
    assert 'lin.weight' in str(e.value)
    assert 'checkpoint (4, 4) vs model (8, 4)' in str(e.value)


def test_a_missing_parameter_is_an_error():
    src, dst = Small(), Small()
    state = src.state_dict()
    del state['lin.bias']
    with pytest.raises(RuntimeError) as e:
        utils.load_matching_state_dict(dst, state, name='small')
    assert 'lin.bias missing' in str(e.value)


def test_buffers_may_change_shape():
    '''Grids are rebuilt from the box size, they legitimately differ when the box changes'''
    src, dst = Small(buffer_size=6), Small(buffer_size=10)
    src.lin.weight.data.fill_(0.5)
    n_params, n_buffers = utils.load_matching_state_dict(dst, src.state_dict(), name='small')
    assert (n_params, n_buffers) == (2, 0)
    assert torch.equal(dst.lin.weight, src.lin.weight)
    assert dst.grid.shape == (10,)


def test_relaxed_load_warns_instead_of_raising():
    src, dst = Small(out=4), Small(out=8)
    n_params, _ = utils.load_matching_state_dict(dst, src.state_dict(), name='small', strict=False)
    assert n_params == 0  # neither the weight nor the bias fits


def test_module_prefix_of_parallel_checkpoints_is_accepted():
    '''DDP and DataParallel insert their own name into every key of the state dict'''
    src, dst = Small(), Small()
    src.lin.weight.data.fill_(0.25)
    wrapped = {'module.' + k: v for k, v in src.state_dict().items()}
    utils.load_matching_state_dict(dst, wrapped, name='small')
    assert torch.equal(dst.lin.weight, src.lin.weight)

    nested = {k.replace('lin', 'lin.module') if k.startswith('lin') else k: v
              for k, v in src.state_dict().items()}
    utils.load_matching_state_dict(Small(), nested, name='small')


def build_decoder(num_bodies):
    masks = None
    if num_bodies:
        masks = dict(com_bodies=torch.randn(num_bodies, 3), in_relatives=torch.randn(num_bodies, 3),
                     rotate_directions=torch.randn(num_bodies, 3),
                     orient_bodies=torch.eye(3).repeat(num_bodies, 1, 1),
                     principal_axes=torch.eye(3).repeat(num_bodies, 1, 1),
                     radii_bodies=torch.rand(num_bodies, 3))
    with contextlib.redirect_stdout(io.StringIO()):
        model = HetOnlyVAE(Lattice(129, extent=0.5), 3, 256, 3, 256, -1, 8, encode_mode='grad',
                           enc_mask=-1, enc_type='vanilla', enc_dim=None, domain='fourier',
                           activation=nn.ReLU, ref_vol=None, Apix=2.0, template_type='conv',
                           warp_type=None, num_struct=1, device='cpu', symm=None, ctf_grid=None,
                           deform_emb_size=2, downfrac=0.5, templateres=64, window_r=0.85,
                           masks_params=masks, num_bodies=num_bodies, z_affine_dim=4)
    return model.decoder


def test_forgetting_num_bodies_is_caught():
    '''eval_vol without --num-bodies leaves the rigid body head at its initial value

    The head is zero initialized, so the bodies simply stop moving and the volumes come out
    looking like a plausible consensus structure.
    '''
    trained = build_decoder(num_bodies=4)
    with pytest.raises(RuntimeError) as e:
        utils.load_matching_state_dict(build_decoder(num_bodies=0), trained.state_dict(),
                                       name='decoder')
    assert 'affine_head' in str(e.value)
    assert '--num-bodies' in str(e.value)


def test_matching_num_bodies_loads_the_whole_decoder():
    trained = build_decoder(num_bodies=4)
    decoder = build_decoder(num_bodies=4)
    n_params, _ = utils.load_matching_state_dict(decoder, trained.state_dict(), name='decoder')
    assert n_params == len(list(decoder.named_parameters()))


def test_load_pkl_falls_back_to_the_cpu(tmp_path, monkeypatch):
    '''Configs written by an older trainer hold Apix as a tensor on the training gpu'''
    path = tmp_path / 'config.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'model_args': {'Apix': torch.tensor(2.2653)}}, f)

    calls = []
    real_load = pickle.load

    def refuses_the_first_call(f, **kwargs):
        # the very first call is the one load_pkl makes, the later ones come from torch.load
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('Attempting to deserialize object on a CUDA device')
        return real_load(f, **kwargs)

    fallbacks = []

    class SpyUnpickler(utils._CpuUnpickler):
        def __init__(self, *args, **kwargs):
            fallbacks.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(utils.pickle, 'load', refuses_the_first_call)
    monkeypatch.setattr(utils, '_CpuUnpickler', SpyUnpickler)
    cfg = utils.load_pkl(str(path))
    assert fallbacks == [1]  # the cpu mapping unpickler took over
    assert float(cfg['model_args']['Apix']) == pytest.approx(2.2653)


def test_load_pkl_still_reads_a_plain_pickle(tmp_path):
    path = tmp_path / 'plain.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'downfrac': 0.75}, f)
    assert utils.load_pkl(str(path)) == {'downfrac': 0.75}
