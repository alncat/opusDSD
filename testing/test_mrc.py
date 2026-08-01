'''
Reading a particle stack the three ways it can be handed to the commands

A stack can be given as an .mrcs, as a .star, or as a .txt listing .mrcs files, and it can be read
eagerly or one image at a time. All of those routes have to return the same pixels.
'''
import os

import numpy as np
import pytest

from cryodrgn import dataset
from cryodrgn import mrc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.fixture(scope='module')
def particles():
    images, _ = mrc.parse_mrc(os.path.join(DATA, 'toy_projections.mrcs'), lazy=False)
    return images


def test_lazy_read_matches_eager_read(particles):
    lazy, _ = mrc.parse_mrc(os.path.join(DATA, 'toy_projections.mrcs'), lazy=True)
    assert np.array_equal(np.asarray([image.get() for image in lazy]), particles)


@pytest.mark.parametrize('stack', ['toy_projections.star', 'toy_projections.txt'])
def test_star_and_txt_resolve_to_the_same_stack(stack, particles):
    loaded = dataset.load_particles(os.path.join(DATA, stack), datadir=DATA)
    assert np.array_equal(loaded, particles)
