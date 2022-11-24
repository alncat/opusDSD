import healpy as hp
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from . import lie_tools
from . import fft
from . import lattice
import numpy as np
import matplotlib.pyplot as plt
import heapq

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, use_fourier=False, mode='bilinear', render_size=180):
        super().__init__()

        self.mode = mode

        # create sampling grid
        #vectors = [torch.arange(0, s) for s in size]

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        self.templateres = size
        self.normalize = normalize
        self.use_fourier = use_fourier
        if use_fourier:
            self.render_size = render_size
            #self.crop_size = min(int(self.render_size*np.sqrt(2)/2+1)*2, self.templateres)
            self.crop_size = self.templateres
            self.xdim = self.render_size//2
            x_idx = torch.arange(self.xdim+1)/float(self.crop_size//2) #[0, s]
            y_idx = torch.arange(-self.xdim, self.xdim)/float(self.crop_size//2) #[-s, s)
            grid  = torch.meshgrid(y_idx, x_idx)
            xgrid = grid[1] #change fast [[0,1,2,3]]
            #ygrid = torch.roll(grid[0], shifts=(self.x_size), dims=(0)) #fft shifted, center at the corner
            ygrid = grid[0]
            zgrid = torch.zeros_like(xgrid)
            grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
            #mask = grid.pow(2).sum(-1) < ((self.xdim-1)/self.templateres*2) ** 2
            print(grid.shape)
            self.register_buffer("gridF", grid)
            self.ori_size = 192
            x_idx = torch.arange(self.xdim+1)/float(self.ori_size//2) #[0, s]
            y_idx = torch.arange(-self.xdim, self.xdim)/float(self.ori_size//2) #[-s, s)
            grid  = torch.meshgrid(y_idx, x_idx)
            xgrid = grid[1] #change fast [[0,1,2,3]]
            #ygrid = torch.roll(grid[0], shifts=(self.x_size), dims=(0)) #fft shifted, center at the corner
            ygrid = grid[0]

            #2d frequency grid
            freqs2d = torch.stack([xgrid, ygrid], dim=-1).unsqueeze(0).unsqueeze(0)/2
            self.register_buffer("freqs2d", freqs2d)

            zgrid = torch.zeros_like(xgrid)
            grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
            self.register_buffer("gridF_ref", grid)

        else:
            if self.normalize:
                zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                    np.linspace(-1., 1., self.templateres),
                                    np.linspace(-1., 1., self.templateres), indexing='ij')
            else:
                zgrid, ygrid, xgrid = np.meshgrid(np.arange(self.templateres),
                                      np.arange(self.templateres),
                                      np.arange(self.templateres), indexing='ij')
            #xgrid is the innermost dimension (-1, ..., 1)
            self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

    def rotate(self, rot):
        return self.grid @ rot

    def sample(self, src):
        return F.grid_sample(src, self.grid, align_corners=ALIGN_CORNERS)

    def pad(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        pad_size = (out_size - src_size)//2
        if pad_size == 0:
            return src
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size))

    def pad_FT(self, src, out_size):
        ypad_size = (out_size - self.render_size)//2
        return F.pad(src, (0, ypad_size, ypad_size, ypad_size-1))

    def rotate_and_sample(self, src, rot):
        pos = self.rotate(rot)
        return F.grid_sample(src, pos, align_corners=ALIGN_CORNERS)

    def fourier_transform(self, template, padding=True):
        if padding:
            pad_size = self.templateres//2
            template = F.pad(template, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size))
        template_FT = fft.torch_rfft3_center(template)
        #template_FT = template_FT[..., 1:, 1:, :self.templateres]
        template_FT = torch.cat((template_FT[..., 1:].flip(dims=(-1,-2,-3)).conj(), template_FT), dim=-1)
        template_FT = F.pad(template_FT, (0, 0, 0, 1, 0, 1))
        return template_FT

    def ctf_correction(self, src, ctf_i):
        size = src.shape[-2]
        src = fft.torch_fft2_center(src)
        ctf_i = torch.fft.fftshift(ctf_i, dim=(-2))
        ctf_i = utils.crop_fft(ctf_i, size)
        ctf_i = torch.fft.fftshift(ctf_i, dim=(-2))
        assert src.shape[-1] == ctf_i.shape[-1] and src.shape[-2] == ctf_i.shape[-2]
        return src*ctf_i

    def find_top_angles(self, y_recon, y, euler_samples):
        y_recon_2 = -2.*y + y_recon
        scale = y_recon.shape[-2] ** 2
        y_recon_2[:, :, 0] *= 0.5
        y_recon_2[:, :, y_recon.shape[-1] - 1] *= 0.5
        l2_diff = (y_recon.conj()*y_recon_2).real.sum(dim=(-1, -2))/scale

        #print(l2_diff*2)
        #y_recon = fft.torch_ifft2_center(y_recon)
        #y = fft.torch_ifft2_center(y)
        #l2_diff = (-2.*y_recon*y + (y_recon*y_recon)).sum(dim=(-1, -2))
        #print(l2_diff)

        probs = F.softmax(-l2_diff.detach(), dim=-1).detach()
        inds_ret = torch.topk(probs, 8, dim=-1)
        inds = inds_ret.indices
        vals = inds_ret.values
        inds = inds.unsqueeze(-1).repeat(1, 3)
        #print(euler_samples, inds)
        top_euler = torch.gather(euler_samples, 0, inds).squeeze(1)
        return top_euler

    def global_2d_search(self, proj_fft, ref_fft, k=4):
        B = proj_fft.size(0)
        proj_fft = proj_fft.unsqueeze(2)
        #print(ref_fft.shape, proj_fft.shape)
        cross_spectrum = (proj_fft * ref_fft.conj())
        cross_spectrum = torch.fft.ifftshift(cross_spectrum, dim=(-2)) #(1, r_t, psi, x, y)
        #cross_spectrum /= cross_spectrum.abs()
        r = fft.torch_ifft2_center(cross_spectrum)
        #print(r.shape)
        width = r.size(-1)
        num_psis = r.size(-3)
        num_eulers = r.size(-4)
        r_k = torch.topk(r.reshape(B, -1), k, dim=-1, sorted=True)
        #r_k_trans = torch.topk(r.view(B*num_eulers*num_psis, -1), k, dim=-1) #(B*num_eulers*num_psis, k)
        ##search optimal psi for each euler
        #r_k_psi = torch.topk(r_k_trans.values.view(B*num_eulers, -1), k, dim=-1) #(B*num_eulers, num_psis*k)
        ##retrieve trans indices
        #r_k_trans_ind = r_k_trans.indices.view(B*num_eulers, -1)[[torch.arange(B*num_eulers*k)//k,
        #                                                         r_k_psi.indices.view(-1)]]
        ##search optimal euler
        #r_k_euler = torch.topk(r_k_psi.values.view(B, -1), k, dim=-1) #(B, num_eulers*k)
        ##retrieve indices
        #r_k_psi_ind = r_k_psi.indices.view(B, -1)[[torch.arange(B*k)//k, r_k_euler.indices.view(-1)]]
        #r_k_trans_ind = r_k_trans_ind.view(B, -1)[[torch.arange(B*k)//k, r_k_euler.indices.view(-1)]]
        ##convert to degree
        #r_k_psi_ind = r_k_psi_ind//k * ang_interval
        #r_k_trans_ind = torch.stack([r_k_trans_ind % width, r_k_trans_ind // width]) - width//2

        r_k_val = r_k.values #mulitply by 2
        r_k_ind = r_k.indices # (B, -1)
        #center = torch.tensor([0., 0., width//2, width//2]).to(r_k_val.get_device())
        #scale = torch.tensor([1., ang_interval, 1., 1.]).to(r_k_val.get_device())
        r_k_ind = torch.stack([r_k_ind // (width*width*num_psis), (r_k_ind // (width*width)) % num_psis,
                               r_k_ind % width, (r_k_ind // width) % width,
                              ], -1)
        plt.imshow(r[0, r_k_ind[0, 0, 0], r_k_ind[0, 0, 1]].detach().cpu())
        #e = r_k_ind[0, 0, 0]
        #p = r_k_ind[0, 0, 1]
        #x = r_k_ind[0, 0, 3]
        #y = r_k_ind[0, 0, 2]
        #print(e, p, x, y)
        #trans = r[0, e, p, x-1:x+2, y-1:y+2]
        #t = r_k_ind[0, 0, 2:4] - width//2
        #t_1 = torch.tensor([x, y]).to(x.get_device()) - width//2
        #ps = self.translate_ft(-t*192/width)
        #ps_1 = self.translate_ft(-t_1*192/width)
        ##print(ps.shape, proj_fft.shape, ref_fft.shape, t, r.shape)
        #head = (self.render_size-width)//2
        #ps = ps[..., head:head+width, :width//2+1]
        #ps_1 = ps_1[..., head:head+width, :width//2+1]
        #corr = self.l2_corr(proj_fft[0, e], ref_fft[p]*ps)
        #print(corr, r[0, e, p, x, y])
        #corr = proj_fft[0, e]*((ref_fft[p]*ps.conj()).conj())
        #print(corr.real.sum()/(90*45), r[0, e, p, 45, 45])
        #print(trans)
        #print(trans.mean(), trans.std())
        #psis = r[0, e, p-1:p+2, x, y]
        #print(psis.mean(), psis.std())
        #es = r[0, e-1:e+2, p, x, y]
        #print(es.mean(), es.std())

        #r_k_ind = r_k_ind.float() - center # (B, k, 2)
        return r_k_val, r_k_ind

    def local_search(self, src, ref, ctf, coords, euler2, hp_order, out_size=None, k=4):
        #get neighborhoods of coords
        #print(coords.shape)
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        euler_pixs = hp.ang2pix(hp_order//2, euler1, euler0, nest=True)
        neighbor_pixs = np.arange(4)[None, :] + 4*euler_pixs[:, None]
        neighbor_pixs = neighbor_pixs.flatten()
        #neighbor_pixs = hp.get_all_neighbours(hp_order, euler1, euler0)
        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order, neighbor_pixs, nest=True)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(src.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(src.get_device())/np.pi*180

        #neighbor_euler0 = torch.tensor(neighbor_euler0.T[:, :]).float().to(src.get_device())/np.pi*180
        #neighbor_euler1 = torch.tensor(neighbor_euler1.T[:, :]).float().to(src.get_device())/np.pi*180

        neighbor_euler0 = torch.cat([coords[:, 0].float(), neighbor_euler0], dim=-1)
        neighbor_euler1 = torch.cat([coords[:, 1].float(), neighbor_euler1], dim=-1)

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(-1, 2) #(s*neighbor, 2)

        uniq_eulers = torch.unique(neighbor_eulers_flatten, dim=0)
        #print(uniq_eulers.shape, uniq_eulers)
        #print("euler2, ", euler2.shape)

        if out_size is not None:
            ctf_c = utils.crop_fft(ctf, out_size)
        else:
            ctf_c = ctf

        with torch.no_grad():
            sampled_H = self.rotate_and_sample_euler(src, uniq_eulers, out_size=out_size) #(1, s*neighbor, H, W)
            rotated_refs, psis = self.sample_ref(ref, hp_order, euler2, [-1, 2], out_size=out_size)

        #print(sampled_H.shape, ctf_c.shape)
        r_k_val, r_k_ind = self.translation_search(sampled_H*ctf_c, rotated_refs, k=k, out_size=12)
        #print(neighbor_eulers, psis)
        r_k_ind_euler = r_k_ind[0, :, 0]
        r_k_ind_psi = r_k_ind[0, :, 1]
        top_eulers = uniq_eulers[r_k_ind_euler]
        #print(top_eulers)
        top_psis = psis[r_k_ind_psi]

        #top_coords = torch.cat([top_eulers, top_psis.unsqueeze(-1)], dim=-1)
        width = rotated_refs.size(-2)
        top_trans = r_k_ind[0, :, 2:].float() - 12//2
        #print(top_coords, top_trans)
        return r_k_val, top_eulers, top_psis, top_trans

    def translate_ft(self, t):
        #t (T, 2)
        coords = self.freqs2d #(1, H, W, 2)
        #print(coords)
        #img = img # Bxhxw
        t = t.unsqueeze(-2).unsqueeze(-1) # Tx1x2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi # TxHxWx1
        tfilt = tfilt.squeeze(-1) # TxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # TxHxW
        s = torch.sin(tfilt) # TxHxW
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))
        return phase_shift

    def sample_ref(self, ref, hp_order, euler2, search_range, out_size=None):
        #ref: (B, 2, h, w)
        #sample 2d rotated refs
        B = euler2.size(0)
        ref = ref.repeat(B, 1, 1, 1)
        ang_interval = hp.max_pixrad(hp_order)*180./np.pi
        rotated_refs = []
        #sample refs in 2d
        start = search_range[0]
        end = search_range[1]
        eulers = []
        for i in np.arange(start, end):
            euler_i = euler2 + i*ang_interval
            rotated_ref = self.rotate_2d(ref, -euler_i, out_size=out_size)
            #print(rotated_ref.shape, euler_i.shape)
            rotated_refs.append(rotated_ref)
            eulers.append(euler_i)
        #rotated_refs: (s, h, w)
        rotated_refs = torch.cat(rotated_refs, dim=0)
        eulers = torch.cat(eulers, dim=0)
        return rotated_refs, eulers

        #w = rotated_refs.size(-2)
        #head = (self.render_size - w)//2
        #rotated_refs *= phase_shifts[..., head:head+w, :w//2+1]
        #shifted_refs = [rotated_refs]
        #psen = torch.tensor([[0.5, 0], [0, 0.5]]).to(trans.get_device())*s_size*self.ori_size/w
        #psen = self.translate_ft(-psen)#(1, 2, H, W)
        #psen = psen.squeeze(0).unsqueeze(1) #(2, 1, H, W)
        #rotated_refs1 = rotated_refs*psen
        #rotated_refs2 = rotated_refs*psen.conj()

    def translation_search(self, proj_fft, ref_fft, out_size=None, k=4):
        B = proj_fft.size(0)
        proj_fft_norm = self.l2_corr(proj_fft, proj_fft).sqrt()
        #print(ref_fft.shape, proj_fft.shape, proj_fft_norm.shape)
        proj_fft /= proj_fft_norm.unsqueeze(-1).unsqueeze(-1)
        proj_fft = proj_fft.unsqueeze(2)
        #ref_fft = ref_fft.unsqueeze(1).unsqueeze(1)
        cross_spectrum = (proj_fft * ref_fft.conj())
        #padding cross_spectrum
        cur_size = ref_fft.size(-2)
        if out_size is not None and out_size > cur_size:
            ypad_size = (out_size - cur_size)//2
            cross_spectrum = F.pad(src, (0, ypad_size, ypad_size, ypad_size))

        cross_spectrum = torch.fft.ifftshift(cross_spectrum, dim=(-2)) #(1, r_t, psi, x, y)
        r = fft.torch_ifft2_center(cross_spectrum)
        #crop r
        head = (cur_size - out_size)//2
        tail = head + out_size
        r = r[..., head:tail, head:tail]
        #print(r.shape)
        width = r.size(-1)
        num_psis = r.size(-3)
        num_eulers = r.size(-4)
        r_k = torch.topk(r.reshape(B, -1), k, dim=-1, sorted=True)
        r_k_val = r_k.values #mulitply by 2
        r_k_ind = r_k.indices # (B, -1)
        r_k_ind = torch.stack([torch.div(r_k_ind, (width*width*num_psis), rounding_mode="floor") % num_eulers,
                               torch.div(r_k_ind, (width*width), rounding_mode="floor") % num_psis,
                               r_k_ind % width,
                               torch.div(r_k_ind, width, rounding_mode="floor") % width,
                              ], -1) #(B, k, 4)
        #plt.imshow(r[0, r_k_ind[0, 0, 0], r_k_ind[0, 0, 1]].detach().cpu())
        return r_k_val, r_k_ind


    def rotate_and_sample_euler(self, src, euler, out_size=None):
        #R = euler2 euler1 (euler0)
        rot = lie_tools.euler_to_SO3(euler).unsqueeze(-3) #(B, 1, 3, 3)
        #print(self.gridF.shape, rot.shape, euler[..., :2])
        if out_size is not None:
            out_xdim = out_size//2 + 1
            head = (self.render_size - out_size)//2
            tail = head + out_size
            gridF = self.gridF[..., head:tail, :out_xdim, :]
        else:
            gridF = self.gridF
        pos = gridF @ rot #(1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        src = torch.view_as_real(src.squeeze(1)).permute([0, 4, 1, 2, 3])
        sampled = F.grid_sample(src, pos, align_corners=True)
        #convert to 2d scale
        scale_factor = self.ori_size ** 2 / self.templateres ** 3
        sampled = torch.view_as_complex(sampled.permute([0, 2, 3, 4, 1]).contiguous())
        sampled_H = sampled*scale_factor

        return sampled_H

    def hp_sample(self, src, euler, ref, ctf, trans=None):
        ori_size = ref.shape[-2]
        coarse_size = 32

        hp_order = 4
        ref = self.transform_ref(ref)
        ang_interval = hp.max_pixrad(hp_order)*180./np.pi
        start_psi = torch.tensor([0.]).to(ref.get_device())
        rotated_refs, psis = self.sample_ref(ref, hp_order, start_psi, [0, 360/ang_interval], out_size=coarse_size)
        #print("rotated_refs: ", rotated_refs.shape, psis.shape)
        #plt.show()

        #ctf = torch.fft.fftshift(ctf, dim=(-2))
        center = self.render_size//2
        #sample this euler
        euler0 = euler[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = euler[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        num_eulers = 12*hp_order**2
        pixs = np.arange(num_eulers)
        euler1s, euler0s = hp.pix2ang(hp_order, pixs)

        euler0s = torch.tensor(euler0s).float().to(euler.get_device())/np.pi*180
        euler1s = torch.tensor(euler1s).float().to(euler.get_device())/np.pi*180
        #sample s2
        eulers = torch.stack([euler0s, euler1s], dim=-1)

        euler_np = np.mod(euler.cpu().numpy(), 360)

        #print("start: ", euler.cpu().numpy(), eulers.shape, trans.cpu().numpy())
        #print(eulers.shape, ctf.shape, src.shape, rotated_refs.shape)
        batch_size = num_eulers
        k = 16
        pq = []
        tot_r_mean = []

        for i in range(0, len(eulers), batch_size):
            eulers_b = eulers[i:i+batch_size, :]
            with torch.no_grad():
                sampled_H = self.rotate_and_sample_euler(src, eulers_b, out_size=coarse_size)
                #print(ctf[...,center, 0], sampled_H[...,center, 0])
                #crop samples and refs
                ctf_c = utils.crop_fft(ctf, coarse_size)
                sampled_H_ctf = sampled_H*ctf_c
                #print(sampled_H_ctf.shape, rotated_refs.shape)
                top_k_val, top_k_ind = self.translation_search(sampled_H_ctf, rotated_refs, k=k, out_size=12)
                top_k_s2_ind = top_k_ind[..., 0].long() #top_k_val, (B, batch_size) (B, batch_size)
                top_k_psi_ind = top_k_ind[..., 1].long()
                top_k_val *= (coarse_size/self.render_size) ** 2
            #print("euler_b: ", eulers_b.cpu().numpy(), r_mean, r_std)
            tot_r_mean.append(top_k_val)
            top_eulers = eulers_b[top_k_s2_ind[0]]
            top_psis   = psis[top_k_psi_ind[0]].unsqueeze(-1)
            #top_trans  = top_k_ind[0, ..., 2:].float() - coarse_size//2
            #print(top_psis.shape, psis.shape, top_eulers.shape)
            top_k_coords = torch.cat((top_eulers, top_psis,), dim=-1)
            #for i_k in range(k):
            #    if len(pq) >= k and pq[0][0] < top_k_val[0, i_k]:
            #        heapq.heappop(pq)
            #    if len(pq) < k:
            #        heapq.heappush(pq, (top_k_val[0, i_k], top_k_coords[i_k]))
            #        #print(pq[0][0].cpu().numpy(), top_k_val[0, i].cpu().numpy(), top_k_coords[i].cpu().numpy(), euler_np)
        tot_r_mean = torch.cat(tot_r_mean, dim=-1)
        #compute normalization factor
        #print("r std: ", tot_r_mean.std(), "mean: ", tot_r_mean.mean())
        #tot_norm = (tot_r_mean).exp().sum(dim=-1)
        #print("tot_norm: ", tot_norm)
        top_coords = []
        #for i in range(k):
        #    top_coords.append(pq[i][1])
            #print("final: ", pq[i][0].cpu().numpy(), pq[i][1].cpu().numpy(), euler_np)
        #top_coords = torch.stack(top_coords, dim=0)
        top_coords = top_k_coords
        #print(top_coords, euler_np)

        top_eulers = top_coords[:, :2]
        top_psis = top_coords[:, 2]

        out_size = coarse_size
        #ctf_c = utils.crop_fft(ctf, out_size)
        #sampled_H = self.rotate_and_sample_euler(src, top_eulers, out_size=out_size)
        #print(self.l2_corr(sampled_H[0,:]*ctf_c, sampled_H[0,:]*ctf_c*128**2)*(out_size/self.render_size)**2*0.5)

        #sample top eulers
        for i in range(1):
            #convert dictionary to coordinates
            hp_order *= 2
            out_size *= 2
            top_eulers = torch.unique(top_eulers, dim=0)
            top_psis = torch.unique(top_psis, dim=0)
            r_k_vals, top_eulers, top_psis, top_trans = self.local_search(src, ref, ctf,
                                                                      top_eulers, top_psis, hp_order, k=k, out_size=out_size)
            r_k_vals *= (out_size/self.render_size) ** 2
        #print(top_eulers, top_psis, r_k_vals)

        out_size = 128
        #sample according to euler
        sampled_H = self.rotate_and_sample_euler(src, top_eulers, out_size=out_size)
        #sampled_H = torch.fft.ifftshift(sampled_H, dim=(-2))
        B = top_psis.size(0)
        ref = ref.repeat(B, 1, 1, 1)
        #rotated_refs = self.rotate_2d(ref, -top_psis, out_size=out_size)
        #shift refs
        #print(trans.cpu().numpy())
        #print(top_trans)
        ps = self.translate_ft(-top_trans*self.ori_size/out_size*2)
        #crop to out_size
        ps = utils.crop_fft(ps, out_size)
        #ctf_c = utils.crop_fft(ctf, out_size)

        rotated_refs = rotated_refs*ps

        #print(ps.shape, rotated_refs.shape, sampled_H.shape)
        #print(self.l2_corr(sampled_H[0,0]*ctf_c, rotated_refs[0,0])*(out_size/self.render_size)**2)
        #print(self.l2_corr(sampled_H[0,:]*ctf_c, sampled_H[0,:]*ctf_c*out_size**2)*(out_size/self.render_size)**2*0.5)

        return sampled_H, rotated_refs

    def hp_sample_(self, src, euler, ref, ctf, trans=None):
        B = ref.size(0) #(B represents different views)
        ori_size = ref.shape[-2]
        out_size  = 128

        #print(ref.shape)
        #ref = self.transform_ref(ref)
        ##print(ref.shape)
        ##euler2 = euler[:, 2]
        ##ang_interval = hp.max_pixrad(16)*180./np.pi
        #rotated_refs = []
        ###for i in torch.arange(-180, 180, ang_interval):
        ###    euler_i = i.to(ref.get_device())
        #euler_i = torch.zeros(B).to(ref.get_device())
        #rotated_ref = self.rotate_2d(ref, -euler_i, out_size=out_size) #euler (B,)
        #rotated_refs.append(rotated_ref)
        #rotated_refs = torch.cat(rotated_refs, dim=0) #(B, H, W)
        rotated_refs = torch.fft.fftshift(ref, dim=-2)
        rotated_refs = utils.crop_fft(rotated_refs, out_size)

        #sample this euler
        #sampled_H = self.rotate_and_sample_euler(src, euler[..., :3], out_size=out_size)
        #r_k_vals, top_eulers, top_psis, top_trans = self.local_search(src, ref, ctf,
        #                                                              euler[..., :2], euler[..., 2:], 16, k=4, out_size=out_size)
        #top_coords = torch.cat([top_eulers, top_psis], dim=-1)
        #print(r_k_vals, top_coords, euler, top_trans*self.ori_size/out_size, trans)
        sampled_H = self.rotate_and_sample_euler(src, euler, out_size=out_size)

        #ps = self.translate_ft(-top_trans*self.ori_size/out_size)
        # crop to out_size
        #ps = self.translate_ft(-trans) #(1, B, H, W)
        #ps = utils.crop_fft(ps, out_size)

        #print(src.shape, euler.shape, sampled_H.shape, rotated_refs.shape, ps.shape)
        #rotated_refs = rotated_refs*ps #(B, H, W) x (1, B, H, W) -> (1, B, H, W)
        #print("rotated_refs: ", rotated_refs.shape, sampled_H.shape)

        return sampled_H, rotated_refs.unsqueeze(0)

    def local_sample(self, src, euler, ref, ctf, trans=None):
        ori_size = ref.shape[-2]
        out_size  = 128

        ref = self.transform_ref(ref)
        euler2 = euler[:, 2]
        ang_interval = hp.max_pixrad(16)*180./np.pi
        rotated_refs = []
        #for i in torch.arange(-180, 180, ang_interval):
        #    euler_i = i.to(ref.get_device())
        euler_i = torch.tensor([0.]).to(ref.get_device())
        rotated_ref = self.rotate_2d(ref, -euler_i, out_size=out_size)
        rotated_refs.append(rotated_ref)
        rotated_refs = torch.cat(rotated_refs, dim=0)

        #center = self.render_size//2
        #sample this euler

        #sampled_H = self.rotate_and_sample_euler(src, euler[..., :3], out_size=out_size)
        #r_k_vals, top_eulers, top_psis, top_trans = self.local_search(src, ref, ctf,
        #                                                              euler[..., :2], euler[..., 2:], 16, k=4, out_size=out_size)
        #sample according to euler
        #print(top_eulers.shape, top_psis.shape)
        #top_coords = torch.cat([top_eulers, top_psis], dim=-1)
        #print(r_k_vals, top_coords, euler, top_trans*self.ori_size/out_size, trans)
        sampled_H = self.rotate_and_sample_euler(src, euler, out_size=out_size)
        #print(top_eulers, top_k_coords)
        #print(euler0, euler1, euler2.cpu().numpy(), neighbor_eulers.shape)
        #ps = self.translate_ft(-top_trans*self.ori_size/out_size)
        ps = self.translate_ft(-trans)
        #crop to out_size
        ps = utils.crop_fft(ps, out_size)

        rotated_refs = rotated_refs*ps

        return sampled_H, rotated_refs

    def rotate_2d(self, ref, euler, out_size=None):
        #euler (B,)
        rot_ref = lie_tools.zrot(euler).unsqueeze(-3) #(B, 1, 3, 3)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        if out_size is not None:
            out_xdim = out_size//2 + 1
            head = (self.render_size - out_size)//2
            tail = head + out_size
            gridF = self.gridF_ref[..., head:tail, :out_xdim, :]
        else:
            gridF = self.gridF_ref

        pos_ref = gridF @ rot_ref
        rotated_ref = F.grid_sample(ref, pos_ref[..., :2].squeeze(0), align_corners=True, mode='bicubic')
        rotated_ref = torch.view_as_complex(rotated_ref.permute([0,2,3,1]).contiguous())
        return rotated_ref

    def transform_ref(self, ref):
        ref = torch.fft.fftshift(ref, dim=-2)
        ref_x_size = ref.shape[-1]
        ref = ref[...,1:,:ref_x_size-1]
        ref = torch.cat([ref[...,:,1:].flip(dims=(-2,-1)).conj(), ref], dim=-1)
        ref = torch.view_as_real(ref).permute([0, 3, 1, 2])
        return ref

    def l2_corr(self, src, tgt, eps=1e-5):
        w = src.size(-2)
        x = w//2
        corr0 = (src[..., 0] * tgt[..., 0].conj()).real
        corr1 = (src[..., 1:x] * tgt[..., 1:x].conj()).real
        corr2 = (src[..., x] * tgt[..., x].conj()).real
        corr = corr0.sum(dim=(-1,-2)) + corr2.sum(dim=(-1,-2)) + corr1.sum(dim=(-1,-2))*2
        #src_norm = src.norm() + eps
        #tgt_norm = tgt.norm() + eps
        return corr/(w*w)

    def rotate_and_sampleFT(self, src, rot):
        pos = self.gridF @ rot
        center = self.render_size//2
        src = torch.view_as_real(src.squeeze(1)).permute([0, 4, 1, 2, 3])
        sampled = F.grid_sample(src, pos, align_corners=True)
        #convert to 2d scale
        scale_factor = self.render_size ** 2 / (self.templateres) ** 3
        sampled = torch.view_as_complex(sampled.permute([0, 2, 3, 4, 1]).contiguous())
        #print(sampled[..., center, center])
        #enforce hermitian symmetry
        #self.check_hermitian_symm(sampled[..., :center].flip(-1).flip(-2), sampled[..., center+1:])
        #self.check_hermitian_symm(sampled[..., center:center+1].flip(-2), sampled[..., center:center+1])
        #sampled_H = sampled[..., :center+1].flip(-1).flip(-2).conj() + sampled[..., center:]
        #sampled_H = sampled[..., :self.render_size, :]*scale_factor
        #sampled_H = sampled[..., :self.render_size, :self.render_size]*scale_factor
        sampled_H = torch.fft.ifftshift(sampled, dim=(-2)) #restore fft order
        #print(sampled_H.shape, sampled_H[..., 0, 0])
        #self.check_hermitian_symm(sampled_H[..., 1:center, :1].flip(-2), sampled_H[..., center+1:, :1])
        return sampled_H

    def check_hermitian_symm(self, x1, x2):
        print(x1, x2)
        diff = (x1 - x2.conj()).abs()/x1.abs()
        print(diff.max())

    def hermitian_symm(self, x):
        x_h = 0.5*(x[..., 1:, :].flip(-2) + x[..., 1:, :])
        x = torch.cat([x[..., :1, :], x_h], dim=-2)
        return x

    def forward(self, src, flow):
        # new locations
        # flow (N, 3, H, W, D)
        shape = flow.shape[2:]
        flow = flow.permute(0, 2, 3, 4, 1)
        new_locs = self.grid + flow

        # need to normalize grid values to [-1, 1] for resampler
        new_locs = 2. * (new_locs/float(self.templateres - 1) - 0.5)
        #for i in range(len(shape)):
        #    new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)

        return F.grid_sample(src, new_locs, align_corners=ALIGN_CORNERS, mode=self.mode)

class HealpixTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, mode='bilinear', render_size=180):
        super().__init__()

        self.mode = mode

        # create sampling grid
        #vectors = [torch.arange(0, s) for s in size]

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        self.templateres = size
        self.normalize = normalize
        self.render_size = render_size
        #self.crop_size = min(int(self.render_size*np.sqrt(2)/2+1)*2, self.templateres)
        self.xdim = self.render_size//2
        x_idx = torch.arange(self.xdim+1)/float(self.templateres//2) #[0, s]
        y_idx = torch.arange(-self.xdim, self.xdim+1)/float(self.templateres//2) #[-s, s]
        grid  = torch.meshgrid(y_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        #ygrid = torch.roll(grid[0], shifts=(self.x_size), dims=(0)) #fft shifted, center at the corner
        ygrid = grid[0]
        zgrid = torch.zeros_like(xgrid)
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
        #mask = grid.pow(2).sum(-1) < ((self.xdim-1)/self.templateres*2) ** 2
        print(grid.shape)
        self.register_buffer("gridF", grid)

        #self.register_buffer("mask", mask)

        #count the number of pixels in each shell
        cubic_grid = torch.meshgrid(y_idx, y_idx, y_idx, indexing='ij')
        cubic_grid = torch.stack([cubic_grid[0], cubic_grid[1], cubic_grid[2]], dim=-1)
        cubic_grid_r = cubic_grid.pow(2).sum(-1).sqrt()
        self.shell_counts = {}
        self.shell_order = {}
        total_counts = 0
        cur_order = 1
        for i in range(self.xdim+1):
            radius = (i)/self.templateres*2
            counts = cubic_grid_r <= radius
            counts_sum = counts.sum()
            self.shell_counts[i] = counts_sum - total_counts
            #order = torch.sqrt(shell_counts[radius]/12)
            hp_num = 12*cur_order**2
            #increase order, can use cur_order til i
            if self.shell_counts[i] > hp_num:
                self.shell_order[cur_order] = i
                cur_order *= 2
            total_counts = counts_sum
        del cubic_grid
        del cubic_grid_r

    def hp_sampling(self, order, template, target):
        sample_shell = self.shell_order[order] - 1
        assert sample_shell > 0
        #rotating with the frequencies inside shell
        sample_grid = self.grid[..., -sample_shell+self.xdim:sample_shell+self.xdim, :sample_shell+1]
        #get all angles of healpix
        num_rot_tilt = 12*order**2
        indices = np.arange(num_rot_tilt)
        angs = hp.pix2ang(order, indices)
        #angs[1] in [0, 360], angs[0] in [0, 180], R=R360 R180 R0
        Rs = utils.R_from_relion_bc(0, angs[0], angs[1])
        sample_grid_Rs = sample_grid @ Rs
        #sample from rotated grid
        projs = F.grid_sample(template, sample_grid_Rs, align_corners=True)
        #compare with the target
        ang_interval = hp.max_pixrad(order)
        B = target.size(0)
        top_indices = []
        diff2s = []
        #get the top k of all difference
        k = 10
        diff2s_k = torch.topk(diff2s.view(B, -1), k, dim=-1, sorted=False)
        diff2s_k_vals = diff2s_k.values
        #convert indices
        diff2s_k_indices = diff2s_k.indices
        diff2s_k_psi = diff2s_k_indices//num_rot_tilt
        diff2s_k_rot_tilt = diff2s_k_indices % num_rot_tilt
        #convert indices to angle

        #search local in finer order

    def psi_search(self, proj_fft, ref_fft, sample_grid, interval, k=4):
        num_psi = 360/ang_interval
        diff2s = []
        trans = []
        B = proj_fft.size(0)
        #ref_fft (B, 1, H, W)
        #proj_fft (B, C, H, W)
        for i in np.arange(0, 360, ang_interval):
            psi = ang_interval*i/180*np.pi
            R_psi = utils.torch_zrot(psi)
            #rotate by psi
            sample_grid_Rs = sample_grid @ R_psi.T
            proj_rots = F.grid_sample(ref_fft, sample_grid_Rs, align_corners=True)
            diff2s_i, trans_i = self.local_transl_search(proj_rots, ref_fft)
            #get top k larget
            #diff2s_i (B, 4)
            diff2s.append(diff2s_i)
            trans.append(trans_i)
        diff2s = torch.stack(diff2s, dim=1) #(B, psi, 4)
        trans  = torch.stack(trans, dim=1).view(B, -1)
        #get top k
        diff2s_k = torch.topk(diff2s.view(B, -1), k, dim=-1, sorted=True)
        #get psi and trans
        k_ind = diff2s_k.indices // 4
        k_psi = k_ind * ang_interval
        k_trans = trans[:, diff2s_k.indices]
        return diff2s_k.values, diff2s_k_psi, diff2s_k_trans


    def local_transl_search(self, proj_fft, ref_fft, k=4):
        B = proj_fft.size(0)
        ref_fft = torch.view_as_complex(ref_fft)
        #print(ref_fft.dtype, proj_fft)
        cross_spectrum = (proj_fft * ref_fft.conj())
        #cross_spectrum /= cross_spectrum.abs()
        r = torch.fft.irfft2(cross_spectrum, dim=(-2, -1))
        width = r.size(-1)
        r_k = torch.topk(r.view(B, -1), k, dim=-1, sorted=True)
        r_k_val = 2.*r_k.values #mulitply by 2
        r_k_ind = r_k.indices # (B, -1)
        shifts = r_k_ind
        #print(r_k_val)
        #r_k_val = F.softmax(r_k_val, dim=-1) #(B, k)
        r_k_ind = torch.stack([r_k_ind % width, r_k_ind // width], -1) - width//2 # (B, k, 2)
        r_k_ind = r_k_ind.float()
        #phase_shifts = self.ctf_grid.translation_grid(r_k_ind) #(B, k, h, w)
        #phase_shifts = torch.sum(phase_shifts*r_k_val.unsqueeze(-1).unsqueeze(-1), axis=1) #(B, h, w)
        #print(weighted_shifts)
        #print(proj_fft[0], ref_fft[0])
        #plt.imshow(r[0].detach().cpu())
        #plt.show()
        return r_k_val, r_k_ind

    def translation_search(self, proj_fft, ref_fft, k=4):
        B = proj_fft.size(0)
        ref_fft = torch.view_as_complex(ref_fft)
        #print(ref_fft.dtype, proj_fft)
        cross_spectrum = (proj_fft * ref_fft.conj())
        #cross_spectrum /= cross_spectrum.abs()
        r = fft.torch_ifft2_center(cross_spectrum)
        width = r.size(-1)
        r_k = torch.topk(r.view(B, -1), k, dim=-1, sorted=False)
        r_k_val = 2.*r_k.values #mulitply by 2
        r_k_ind = r_k.indices # (B, -1)
        #print(r_k_val)
        r_k_val = F.softmax(r_k_val, dim=-1) #(B, k)
        r_k_ind = torch.stack([r_k_ind % width, r_k_ind // width], -1) - width//2 # (B, k, 2)
        r_k_ind = r_k_ind.float()
        phase_shifts = self.ctf_grid.translation_grid(r_k_ind) #(B, k, h, w)
        phase_shifts = torch.sum(phase_shifts*r_k_val.unsqueeze(-1).unsqueeze(-1), axis=1) #(B, h, w)
        #weighted_shifts = torch.sum(r_k_val.unsqueeze(-1)*r_k_ind, axis=1)
        #print(weighted_shifts)
        #print(proj_fft[0], ref_fft[0])
        #plt.imshow(r[0].detach().cpu())
        #plt.show()
        return phase_shifts

    def sample_local_rots(self, src, order, euler):
        neighbor_pix = hp.get_all_neighbors(order, euler[0], euler[1])
        #convert to euler angles
        neighbor_tilts, neighbor_rots = hp.pix2ang(order, neighbor_pix)
        #convert to rotation matrices
        neighbor_Rs = utils.R_from_relion_bc(0, neighbor_tilts, neighbor_rots)
        neighbor_Rs = torch.tensor(neighbor_Rs).to(self.gridF.get_device()).unsqueeze(1)
        #gridF (1, 1, H, W, 3), neighbor_Rs should be (D, 1, 3, 3)
        src = torch.view_as_real(src.squeeze(1)).permute([0, 4, 1, 2, 3]) #(1, C, D, H, W)
        #sample psi
        ang_interval = hp.max_pixrad(order)
        for i in np.arange(-1, 2):
            psi = ang_interval*i + euler[2]
            psi = psi/180*np.pi
            R = utils.zrot(psi)
            #rotate matrices
            Rs = neighbor_Rs @ R
            pos = self.gridF @ Rs #(1, D, H, W, 3)
            sampled = F.grid_sample(src, pos, align_corners=True)
            #translational search

        scale_factor = np.sqrt(self.templateres**3) / self.render_size
        sampled = torch.view_as_complex(sampled.permute([0, 2, 3, 4, 1]).contiguous())
        sampled_H = sampled[..., :self.render_size, :]*scale_factor
        return sampled_H

    def rotate_and_sampleFT(self, src, rot, image_ref):
        #Fimage_ref = torch.fft.fftshift(torch.fft.fft2(image_ref, dim=(-2, -1)), dim=(-2,-1))
        #print(Fimage_ref.shape, src.shape)
        #diff = Fimage_ref - src[..., self.templateres//2, :, :].squeeze(0)
        #diff = diff.abs()/Fimage_ref.abs()
        #print(diff.max())
        #pos = self.rotate(rot)
        pos = self.gridF @ rot
        #neg_pos = -pos
        #pos = torch.where(pos[...,0:1] >= 0, pos, neg_pos)
        #rescale pos[0] from [0, 0.5] to [-1, 1] and pos[1, 2] from [-0.5, 0.5]
        #pos = (pos - self.shift) * self.scale
        #pad src
        #src = F.pad(src, (0, 1, 0, 1, 0, 1))
        center = self.render_size//2
        #pad = (self.templateres - self.render_size)//2
        #sampled = src[..., self.templateres//2, pad:self.render_size+pad+1, pad:self.render_size+pad+1]
        #print(sampled[..., center, center])
        #print(src[..., self.templateres//2, self.templateres//2, self.templateres//2])
        #src = src[..., 1:, 1:, 1:]
        src = torch.view_as_real(src.squeeze(1)).permute([0, 4, 1, 2, 3])
        #print(sampled.shape)
        sampled = F.grid_sample(src, pos, align_corners=True)
        #sampled = src[..., self.templateres//2-1:self.templateres//2, pad-1:self.render_size+pad, pad-1:self.render_size+pad]
        #sampled = sampled[..., center:center+1, :, :]
        #print(pos[..., center:center+1, :, :, :])
        #print(sampled.shape)
        #convert to 2d scale
        scale_factor = self.render_size**2 / self.templateres**3
        sampled = torch.view_as_complex(sampled.permute([0, 2, 3, 4, 1]).contiguous())
        #print(self.mask.shape, sampled.shape)
        #sampled = sampled*self.mask
        #print(sampled[..., center, center])
        #enforce hermitian symmetry
        #self.check_hermitian_symm(sampled[..., :center].flip(-1).flip(-2), sampled[..., center+1:])
        #self.check_hermitian_symm(sampled[..., center:center+1].flip(-2), sampled[..., center:center+1])
        #sampled_H = sampled[..., :center+1].flip(-1).flip(-2).conj() + sampled[..., center:]
        sampled_H = sampled[..., :self.render_size, :]*scale_factor
        #sampled_H = sampled[..., :self.render_size, :self.render_size]*scale_factor
        sampled_H = torch.fft.fftshift(sampled_H, dim=(-2))
        #print(sampled_H[..., 0, 0])
        #self.check_hermitian_symm(sampled_H[..., 1:center, :1].flip(-2), sampled_H[..., center+1:, :1])
        #print(pos.shape, src.shape, sampled_H.shape)
        return sampled_H

    def check_hermitian_symm(self, x1, x2):
        print(x1, x2)
        diff = (x1 - x2.conj()).abs()/x1.abs()
        print(diff.max())

    def hermitian_symm(self, x):
        x_h = 0.5*(x[..., 1:, :].flip(-2) + x[..., 1:, :])
        x = torch.cat([x[..., :1, :], x_h], dim=-2)
        return x

