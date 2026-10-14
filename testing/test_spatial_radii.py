"""Spatial radius semantics without depending on optional motion extensions."""
import contextlib
import io

import pytest
import torch
from torch import nn

from cryodrgn import models, mrc, utils


class SmallTemplate(nn.Module):
    """Keep the actual decoder/grid/export path; replace only the expensive CNN."""
    def __init__(self, in_dim, templateres, num_bodies, z_affine_dim, **kwargs):
        super().__init__()
        self.zdim, self.nb = in_dim, num_bodies
        self.head = nn.Linear(z_affine_dim, (num_bodies+1)*6)
        c = torch.linspace(-1, 1, templateres)
        zz, yy, xx = torch.meshgrid(c, c, c, indexing='ij')
        self.register_buffer('volume', torch.exp(-((xx-.3)**2+yy**2+zz**2)/.04)[None,None])

    def forward(self, z, **kwargs):
        a = self.head(z[:,self.zdim:]).view(-1,self.nb+1,6)
        affine = (torch.cat([torch.ones_like(a[...,:1])*16,a[...,:3]],-1),a[...,3:],None)
        return self.volume.expand(len(z),-1,-1,-1,-1), affine


@pytest.fixture
def make_decoder(monkeypatch):
    monkeypatch.setattr(models,'ConvTemplate',SmallTemplate)

    def build(kind=None):
        geometry = dict(com_bodies=torch.tensor([[4.,0.,0.],[-4.,0.,0.]]),
                        in_relatives=torch.zeros(2,3),
                        rotate_directions=torch.tensor([[4.,0.,0.],[-4.,0.,0.]]),
                        orient_bodies=torch.eye(3).repeat(2,1,1),
                        principal_axes=torch.eye(3).repeat(2,1,1),
                        radii_bodies=torch.full((2,3),6.))
        if kind is not None:
            geometry['radii_kind'] = kind
        with contextlib.redirect_stdout(io.StringIO()):
            dec = models.get_decoder(5,33,1,8,'hartley','vanilla',template_type='conv',
                                     templateres=16,render_size=16,down_vol_size=16,
                                     z_affine_dim=2,masks_params=geometry,fixed_deform=True,
                                     ref_vol=torch.ones(32,32,32))
        dec.get_particle_hopfs = lambda *args, **kwargs: torch.zeros(1,2)
        return dec.eval()
    return build


def test_legacy_inertia_radii_convert_without_mutation():
    old = torch.tensor([[5.,10.,13.]]).sqrt()
    geometry = dict(radii_bodies=old.clone())
    torch.testing.assert_close(utils.spatial_body_radii(geometry),torch.tensor([[3.,2.,1.]]))
    torch.testing.assert_close(geometry['radii_bodies'],old)


def test_tagged_spatial_radii_are_not_converted_twice():
    r = torch.tensor([[3.,2.,1.]])
    torch.testing.assert_close(utils.spatial_body_radii(dict(radii_bodies=r,radii_kind='spatial_rms')),r)


@pytest.mark.parametrize('r,kind',[([1.,1.,4.],'inertia_rms'),
                                  ([1.,0.,2.],'spatial_rms'),
                                  ([1.,float('nan'),2.],'spatial_rms'),
                                  ([1.,2.,3.],'unknown')])
def test_invalid_geometry_is_rejected(r,kind):
    with pytest.raises(ValueError):
        utils.spatial_body_radii(dict(radii_bodies=torch.tensor([r]),radii_kind=kind))


@pytest.mark.parametrize('kind,expected',[(None,18.**.5),('spatial_rms',6.)])
def test_decoder_uses_spatial_radii_in_source_box_units(make_decoder,kind,expected):
    dec = make_decoder(kind)
    torch.testing.assert_close(dec.radius,torch.full((2,3),expected/32))
    assert dec._spatial_blend_radii.item()==1


@pytest.mark.parametrize('matching',[False,True])
def test_old_checkpoint_keeps_radius_and_legacy_tag(make_decoder,matching):
    source = make_decoder()
    state = source.state_dict()
    del state['_spatial_blend_radii']
    state['radius'] = torch.full_like(source.radius,.125)
    target = make_decoder()
    if matching:
        utils.load_matching_state_dict(target,{'module.'+k:v for k,v in state.items()})
    else:
        target.load_state_dict(state)
    torch.testing.assert_close(target.radius,state['radius'])
    assert target._spatial_blend_radii.item()==0
    again = make_decoder()
    utils.load_matching_state_dict(again,target.state_dict())
    assert again._spatial_blend_radii.item()==0
    torch.testing.assert_close(again.radius,state['radius'])


def test_corrected_checkpoint_roundtrip(make_decoder):
    source,target = make_decoder(),make_decoder('spatial_rms')
    utils.load_matching_state_dict(target,source.state_dict())
    torch.testing.assert_close(target.radius,source.radius)
    assert target._spatial_blend_radii.item()==1


def test_old_checkpoint_preserves_render_and_export(make_decoder,tmp_path):
    source = make_decoder()
    source.radius.fill_(6./32)
    source._spatial_blend_radii.zero_()
    state = {k:v for k,v in source.state_dict().items() if k!='_spatial_blend_radii'}
    target = make_decoder()
    utils.load_matching_state_dict(target,state)
    z = torch.randn(2,4)
    def render(dec):
        torch.manual_seed(31)
        return dec(torch.eye(3).unsqueeze(0).expand(2,-1,-1),torch.zeros(2,2),z,
                   euler=torch.zeros(2,3),ref_fft=torch.zeros(2,1,16,16),
                   ctf=torch.ones(2,16,9))['y_recon']
    with torch.no_grad():
        torch.testing.assert_close(render(source),render(target),atol=0,rtol=0)
        for name,dec in [('source',source),('target',target)]:
            dec.save(str(tmp_path/name),z=z[:1],Apix=2.)
    a,_ = mrc.parse_mrc(str(tmp_path/'source.mrc'))
    b,_ = mrc.parse_mrc(str(tmp_path/'target.mrc'))
    assert (a==b).all()
