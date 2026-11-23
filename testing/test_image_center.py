import torch

def _rel(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(((a - b)**2).mean().sqrt()/max(float(a.std()), 1e-9))

def test_particle_image_origin_matches_projection_at_every_inplane_draw():
    from cryodrgn import fft, utils
    from cryodrgn.models import Encoder

    # Sample an analytic two-blob projection independently on the raw detector
    # (FFT origin) and render grid (geometric midpoint). Include a nonzero STAR
    # shift and a crop after Fourier downsampling, as in training.
    raw_size, render, crop = 96, 72, 56
    enc = Encoder(2, raw_size + 1, crop, render_size=render, rank=1,
                  image_center='grid')
    trans = torch.tensor([[1.25, -2.5]])
    def phantom(x, y):
        return torch.exp(-((x-4)**2+(y+2)**2)/18) + .7*torch.exp(-((x+6)**2+(y-3)**2)/12)
    raw = (torch.arange(raw_size) - raw_size//2) * (render/raw_size)
    y, x = torch.meshgrid(raw, raw, indexing='ij')
    image = phantom(x+trans[0, 0]*render/raw_size, y+trans[0, 1]*render/raw_size)[None, None]
    r = torch.arange(crop) - (crop-1)/2
    y, x = torch.meshgrid(r, r, indexing='ij')
    expected = phantom(x, y)[None, None]
    def centre(img):
        return torch.stack([(img * coord).sum()/img.sum() for coord in (x, y)])
    def process():
        return utils.crop_image(fft.torch_ifft2_center(enc.particle_fft(image, trans),
                                                      center_fft=True), crop)
    corrected = process()
    for angle in (0., 45., 90., 135., 180., 270.):
        rot = torch.tensor([angle])
        a = enc.transformer_e.rotate_2d(corrected, rot)
        b = enc.transformer_e.rotate_2d(expected, rot)
        assert (centre(a) - centre(b)).norm() < .02
        assert _rel(a, b) < .001
    enc.image_center = 'fft'
    torch.testing.assert_close(centre(process()) - centre(expected),
                               torch.full((2,), .5), atol=.002, rtol=0)

def test_training_center_follows_checkpoint_unless_explicitly_overridden():
    from cryodrgn.commands.train_multi_dist import resolve_image_center
    assert resolve_image_center(None, 'grad', None) == 'grid'
    assert resolve_image_center(None, 'grad', {'epoch': 17}) == 'fft'
    assert resolve_image_center(None, 'grad', {'image_center': 'grid'}) == 'grid'
    assert resolve_image_center('grid', 'grad', {'epoch': 17}) == 'grid'
    assert resolve_image_center('fft', 'grad', None) == 'fft'
