'''
Phase flipping the images instead of applying the signed ctf to the projection

Both forms describe the same least squares problem, sign(ctf)**2 is one wherever ctf is not zero,
so the flip has to leave the images with the energy they came in with. It does not if the sign is
written straight onto the half spectrum rfft2 produces.
'''
import contextlib
import io

import pytest
import torch

from cryodrgn import ctf as ctfmod
from cryodrgn import fft
from cryodrgn.lattice import CTFGrid

D = 64  # image size
# dfu, dfv, dfang, volt, cs, w, phase_shift, of the spliceosome dataset
CTF_PARAMS = [15000., 15200., 30., 300., 2.7, 0.1, 0.]


@pytest.fixture(scope='module')
def c():
    '''The ctf of a small batch, laid out the way run_batch builds it'''
    with contextlib.redirect_stdout(io.StringIO()):  # CTFGrid logs the grid it built
        grid = CTFGrid(D + 1, 'cpu')
    freqs = grid.freqs2d.view(-1, 2).unsqueeze(0)/1.699
    params = torch.tensor([CTF_PARAMS]).repeat(2, 1)
    return ctfmod.compute_ctf(freqs, *torch.split(params, 1, 1), bfactor=3.75).view(2, D, -1)


def images():
    torch.manual_seed(0)
    return torch.randn(2, 1, D, D)


def test_flipping_keeps_the_energy_of_the_images(c):
    y = images()
    flipped, _ = ctfmod.phase_flip(y, c)
    assert (flipped**2).sum() == pytest.approx((y**2).sum(), rel=1e-5)


def test_flipping_twice_returns_the_images(c):
    y = images()
    once, _ = ctfmod.phase_flip(y, c)
    twice, _ = ctfmod.phase_flip(once, c)
    assert (twice - y).abs().max() < 1e-4


def test_the_returned_ctf_has_no_sign(c):
    _, flipped_c = ctfmod.phase_flip(images(), c)
    assert (flipped_c >= 0).all()
    assert torch.allclose(flipped_c, c.abs())


def test_writing_the_sign_straight_onto_the_half_spectrum_loses_energy(c):
    '''What hermitian_sign is there to prevent, x=0 and x=nyquist are stored only once'''
    y = images()
    naive = fft.torch_ifft2_center(fft.torch_fft2_center(y)*torch.sign(c).unsqueeze(1))
    assert abs((naive**2).sum() - (y**2).sum())/(y**2).sum() > 1e-4


def test_the_residual_is_the_one_of_the_signed_form(c):
    '''||abs(c) P - flip(y)|| and ||c P - y|| are the same, up to a real space mask'''
    torch.manual_seed(1)
    projection = torch.randn(2, 1, D, D)
    y = images()

    def apply(p, kernel):
        return fft.torch_ifft2_center(fft.torch_fft2_center(p)*kernel.unsqueeze(1))

    signed = ((apply(projection, c) - y)**2).sum()
    flipped_y, flipped_c = ctfmod.phase_flip(y, c)
    flipped = ((apply(projection, flipped_c) - flipped_y)**2).sum()
    assert flipped == pytest.approx(signed, rel=1e-3)
