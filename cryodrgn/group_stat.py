import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from cryodrgn.lattice import CTFGrid
import matplotlib.pyplot as plt

from . import lie_tools
from . import utils
log = utils.log

class GroupStat(nn.Module):
    def __init__(self, group, device, D=None, vol_size=None, mu=0.9, group_stats=None, optimize_b=True):
        super(GroupStat, self).__init__()
        num_groups = set(group)
        self.min_group_no = int(min(num_groups))
        self.max_group_no = int(max(num_groups))
        log("min_group_no: {}, max_group_no: {}".format(self.min_group_no, self.max_group_no))

        group = torch.tensor(group).type(torch.int64)
        self.group = group
        self.D = D
        self.vol_size = vol_size
        self.optimize_b = optimize_b
        log("optimize b factor: {}".format(self.optimize_b))

        self.mu = mu
        init_var = 1.#(self.D-1)**2

        if group_stats is not None:
            self.register_buffer('group_variances'        , group_stats['group_variances'])
            self.register_buffer('group_exp_ref_spectrum' , group_stats['group_exp_ref_spectrum'])
            self.register_buffer('group_exp_spectrum'     , group_stats['group_exp_spectrum'])
            self.register_buffer('group_ref_spectrum'     , group_stats['group_ref_spectrum'])
            self.register_buffer('group_ctf_spectrum'     , group_stats['group_ctf_spectrum'])
            self.register_buffer('group_counts'           , group_stats['group_counts'])
            self.register_buffer('group_scales'           , group_stats['group_scales'])
            if self.optimize_b:
                self.group_b_factors = nn.Parameter(group_stats['group_b_factors'])
        else:
            self.register_buffer('group_exp_ref_spectrum', torch.zeros(self.max_group_no + 1, self.vol_size))
            self.register_buffer('group_variances',        torch.ones(self.max_group_no + 1, self.vol_size)*init_var)
            self.register_buffer('group_exp_spectrum',     torch.zeros(self.max_group_no + 1, self.vol_size))
            self.register_buffer('group_ref_spectrum',     torch.zeros(self.max_group_no + 1, self.vol_size))
            self.register_buffer('group_ctf_spectrum',     torch.zeros(self.max_group_no + 1, self.vol_size))
            self.register_buffer('group_counts',           torch.zeros(self.max_group_no + 1,))
            self.register_buffer('group_scales',           torch.zeros(self.max_group_no + 1,))
            if self.optimize_b:
                #note that b = 8*pi^2*(sigmax^2 + sigmay^2)/2
                b_fact = 4*((self.vol_size)/(self.D - 1))**2
                self.group_b_factors = nn.Parameter(data= b_fact *np.pi ** 2 *torch.ones(self.max_group_no+1,))
        self.x_dim = (D-1)//2 + 1
        self.ctf_grid = CTFGrid(self.vol_size+1, device, center=True)
        self.max_r = self.ctf_grid.max_r + 1

    def shell_to_grid(self, x):
        return self.ctf_grid.shell_to_grid(x)

    def get_spectrum(self, x, avg=True, sqr=True, eps=1e-5):
        if sqr:
            x = (x).abs()
            x = x ** 2
        x *= self.ctf_grid.shells_weight
        x = self.ctf_grid.get_shell_sum(x)
        if avg:
            if x.is_cuda:
                x = x/(self.ctf_grid.shells_count.to(x.get_device()) + eps)
            else:
                x = x/(self.ctf_grid.shells_count + eps)
        return x

    def average_shell(self, x, eps=1e-5):
        if x.is_cuda:
            x /= (self.ctf_grid.shells_count.to(x.get_device()) + eps)
        else:
            x /= (self.ctf_grid.shells_count + eps)
        return x

    def get_fsc(self, reference, experimental, grp_vars, mask, ind=None, eps=1e-5, optimize_b=False):
        corr = (-2.*experimental.conj()*reference).real
        if optimize_b:
            corr = self.apply_b_factor(corr, ind)
        corr = self.get_spectrum(corr, sqr=False)[:, :, :mask]
        #corr = torch.mean(corr, dim=1)
        ref_spec = self.get_spectrum(reference)[:, :, :mask]
        exp_spec = self.get_spectrum(experimental)[:, :, :mask]
        #print(corr.shape, ref_spec.shape, grp_vars.shape)
        #ref_spec = torch.mean(ref_spec, dim=1)
        corr /= torch.sqrt(ref_spec*exp_spec + eps)
        return corr

    def get_l2_loss(self, reference, experimental, grp_vars, mask, ind=None, optimize_b=False):
        if optimize_b:
            reference = self.apply_b_factor(reference, ind)
        diff = -2.*experimental*reference.conj() # + reference
        #diff *= reference.conj()
        diff += reference.conj()*reference
        gen_loss = self.get_spectrum(diff.real, avg=False, sqr=False)[:, :, :mask]
        #grp_variance = self.group_variances[group_stat.get_group_ids(ind), :mask] + 1e-5
        #gen_loss = torch.mean(gen_loss, dim=1)
        gen_loss /= grp_vars.unsqueeze(1)
        return gen_loss

    def get_sqr_loss(self, reference, experimental, grp_vars, mask, ind=None, optimize_b=False):
        if optimize_b:
            reference = self.apply_b_factor(reference, ind)
        diff = reference - experimental
        diff = diff*diff.conj()
        gen_loss = self.get_spectrum(diff.real, avg=False, sqr=False)[:, :, :mask]
        #gen_loss /= grp_vars.unsqueeze(1)
        return gen_loss

    def get_grad_loss(self, reference, experimental, grp_vars, mask):
        reference = self.apply_gradient(reference)
        experimental = self.apply_gradient(experimental)
        diff = -2.*experimental + reference
        print(diff.shape)
        diff *= reference.conj()
        gen_loss = self.get_spectrum(diff.real.sum(-1), avg=False, sqr=False)[:, :, :mask]
        #grp_variance = self.group_variances[group_stat.get_group_ids(ind), :mask] + 1e-5
        #gen_loss = torch.mean(gen_loss, dim=1)
        gen_loss /= grp_vars.unsqueeze(1)
        return gen_loss

    def get_em_l2_loss(self, reference, experimental, variance, mask, ind=None, probs=None):
        C = reference.shape[1]
        if self.optimize_b:
            reference = self.apply_b_factor(reference, ind)
        #gen_loss = self.get_spectrum(diff.real, avg=False, sqr=False)[:, :, :mask] #(N, C, r)
        exp_spectrum = self.get_spectrum(experimental, avg=False)
        exp_ref = (reference.conj()*experimental).real
        exp_ref = self.get_spectrum(exp_ref, avg=False, sqr=False)
        ref_spectrum = self.get_spectrum(reference, avg=False)

        #variance = ref_spectrum - 2.*exp_ref

        #now calculate the em probs, since only half of FT is involved in computation,
        #we should multiply them by 2
        #gen_loss = variance[:, :, :mask]
        #print(exp_spectrum.shape, gen_loss.shape, grp_vars.shape)
        #gen_loss = torch.sum(gen_loss/grp_vars.unsqueeze(-2), dim=-1) #(N, C)

        with torch.no_grad():
            #print(gen_loss)
            #variance = variance + exp_spectrum
            #probs = F.softmax(-gen_loss, dim=-1).unsqueeze(-1)
            probs = 1.
            #print(probs, exp_spectrum.shape)

            exp_spectrum = (probs*exp_spectrum).sum(-2)
            exp_ref      = (probs*exp_ref).sum(-2)
            ref_spectrum = (probs*ref_spectrum).sum(-2)
            variance     = (variance).sum(dim=(-1))/(C*self.vol_size **2)

            exp_spectrum = self.average_shell(exp_spectrum)
            exp_ref      = self.average_shell(exp_ref)
            ref_spectrum = self.average_shell(ref_spectrum)
            count        = C #probs.sum(-2).squeeze(-1) #(N, C, 1) -> (N)
            #variance     = self.average_shell(variance)
            self.add_variance(ind, variance, exp_ref, exp_spectrum, ref_spectrum, ctf_spectrum=None, counts=count)

        #gen_loss = probs.squeeze(-1)*gen_loss #(N, C, 1) -> (N, C), C represents different views
        #update variance
        #return gen_loss

    def get_weighted_l2_loss(self, reference, experimental, grp_vars, probs, mask, ind=None):
        if self.optimize_b:
            reference = self.apply_b_factor(reference, ind)
        diff = -2.*experimental + reference
        diff *= reference.conj()
        #print(diff.shape, experimental.shape)
        gen_loss = self.get_spectrum(diff.real, avg=False, sqr=False)[:, :, :mask] #(N, C, r)
        gen_loss /= grp_vars.unsqueeze(1)
        #now calculate the em probs, since only half of FT is involved in computation,
        #we should multiply them by 2
        gen_loss = torch.sum(gen_loss, dim=-1)
        gen_loss = probs*gen_loss
        return gen_loss

    def fit_b_factor(self, ind, mask, eps=1e-5):
        #ref_spec = self.group_ref_spectrum[self.get_group_ids(ind)]
        #exp_ref_spec = self.group_exp_ref_spectrum[self.get_group_ids(ind)]
        ref_spec = self.group_ref_spectrum[ind]
        exp_ref_spec = self.group_exp_ref_spectrum[ind]
        s2 = self.ctf_grid.shells_s2[:mask]
        log_spec = torch.log(exp_ref_spec[:mask].abs() + eps) - torch.log(ref_spec[:mask] + eps)
        avg_s2 = s2.mean()
        avg_log_spec = log_spec.mean(-1, keepdim=True)
        diff_s2 = s2 - avg_s2
        bs = ((log_spec - avg_log_spec)*diff_s2).sum(-1)/(diff_s2 ** 2).sum(-1)
        return bs, bs*s2

    def apply_b_factor(self, src, ind):
        s2 = self.ctf_grid.s2
        bfactor = self.group_b_factors[self.get_group_ids(ind)].view(-1, 1, 1, 1)
        #print(bfactor)
        bfactor = torch.exp(-bfactor/4*s2)
        #print(src.shape, s2.shape, bfactor.shape)
        src = src*bfactor
        return src

    #def get_exp_factor(self, max_r, slope):
    #    s = (self.ctf_grid.s2 + 1e-6).sqrt()
    #    bfactor = torch.exp(-(max_r-s)/slope)
    #    return bfactor

    def crop_ft(self, src, out_size):
        assert len(src.shape) == 4
        Y = src.shape[2]
        assert out_size <= Y//2
        out_size = 100
        out_xdim = out_size//2 + 1
        head = (Y - out_size)//2
        tail = head + out_size
        src_c = src[..., head:tail, :out_xdim]

        #since the foward transform is multiplied with Y**2
        scale_factor = (2*mask/Y)**2
        return cropped*scale_factor

    def downsample(self, src, down_size):
        mask = down_size//2
        assert down_size % 2 == 0
        cropped = self.crop_ft(src)
        down_src = utils.torch_ifft2_center(cropped)
        return down_src

    def apply_gradient(self, src):
        grad_src = src.unsqueeze(-1)*self.ctf_grid.freqs2d*2.*np.pi
        return grad_src

    def get_mask_sum(self, mask):
        return self.ctf_grid.shells_count[:mask].sum()

    def get_group_ids(self, ind):
        return self.group[ind]

    def get_group_scales(self, ind):
        return self.group_scales[self.get_group_ids(ind)]

    def moving_average(self, x, update, mu, count=None):
        #if update.is_cuda:
        #    x = x.to(update.get_device())
        #update = update.cpu()
        if x.is_cuda and not update.is_cuda:
            update = update.to(x.get_device())
        x *= mu
        x += update
        if count is not None:
            x /= count
        return x

    def get_ssnr(self, ind, eps=1e-8):
        grp_vars = self.group_variances[self.get_group_ids(ind)]
        grp_ss   = self.group_exp_ref_spectrum[self.get_group_ids(ind)].abs()
        return grp_ss/(grp_vars + eps)

    def get_wiener_filter(self, ind, eps=1e-8):
        ssnr = self.get_ssnr(ind, eps)
        grp_ctf = self.group_ctf_spectrum[self.get_group_ids(ind)]
        filt = ssnr/(grp_ctf*ssnr + 16.)
        return filt

    def add_variance(self, ind, variance, exp_ref, exp_spectrum, ref_spectrum, ctf_spectrum=None, counts=1.):
        group_ids = self.group[ind]
        for i in range(len(ind)):
            g_id = group_ids[i]
            self.group_counts[g_id] *= self.mu
            self.group_counts[g_id] += counts
            #i_mu = min(self.mu, (self.group_counts[g_id] - 1.)/self.group_counts[g_id])
            i_mu = self.mu
            #count = self.group_counts[g_id]

            self.group_variances[g_id] = self.group_variances[g_id]*i_mu + (1. - i_mu)*variance[i]
            self.group_exp_ref_spectrum[g_id] = self.moving_average(self.group_exp_ref_spectrum[g_id], exp_ref[i], i_mu)
            self.group_exp_spectrum[g_id] = self.moving_average(self.group_exp_spectrum[g_id], exp_spectrum[i], i_mu)
            self.group_ref_spectrum[g_id] = self.moving_average(self.group_ref_spectrum[g_id], ref_spectrum[i], i_mu)
            #self.group_variances[g_id][:70] = (self.group_exp_spectrum[g_id] + self.group_ref_spectrum[g_id] - 2.*self.group_exp_ref_spectrum[g_id])[:70]
            if ctf_spectrum is not None:
                self.group_ctf_spectrum[g_id] = self.moving_average(self.group_ctf_spectrum[g_id], ctf_spectrum[i], i_mu)

    def update_variance(self, ind, reference, experimental, ctf=None, eps=1e-5):
        # reference is the FT of reference projection,
        # experimental is the FT of experimental image
        group_ids = self.group[ind]
        #variance = (reference - experimental).abs()
        #variance = variance ** 2
        #variance = self.ctf_grid.get_shell_sum(variance)/self.ctf_grid.shells_count
        #if reference.is_cuda:
        #    exp_ref = self.ctf_grid.get_shell_sum((reference.conj()*experimental).real) / (self.ctf_grid.shells_count.to(reference.get_device()) + eps)
        #else:
        #    exp_ref = self.ctf_grid.get_shell_sum((reference.conj()*experimental).real) / (self.ctf_grid.shells_count + eps)
        #print(reference.shape, experimental.shape, ctf.shape)

        #exp_spectrum = (experimental).abs()
        #exp_spectrum = exp_spectrum ** 2
        #exp_spectrum = self.ctf_grid.get_shell_sum(exp_spectrum)/self.ctf_grid.shells_count
        exp_spectrum = self.get_spectrum(experimental)
        exp_spectrum = torch.mean(exp_spectrum, dim=1)
        #exp_spectrum = exp_spectrum.squeeze(1)


        with torch.no_grad():
            if self.optimize_b:
                reference = self.apply_b_factor(reference, ind)
            exp_ref = (reference.conj()*experimental).real
            exp_ref = self.get_spectrum(exp_ref, sqr=False)
            exp_ref = torch.mean(exp_ref, dim=1)

            ref_spectrum = self.get_spectrum(reference)
            ref_spectrum = torch.mean(ref_spectrum, dim=1)

            #print(exp_spectrum.shape, ref_spectrum.shape, exp_ref.shape)
            variance = exp_spectrum + ref_spectrum - 2.*exp_ref
            #variance_ = self.get_spectrum(experimental - reference)
            #diff = (variance - variance_).abs()/(variance_ + 1e-5)
            #bool_mask = ( diff > 1e-5)

            #print(diff.max())
            #assert torch.allclose(variance, variance_, rtol=1e-4)

            if ctf is not None:
                ctf_spectrum = self.get_spectrum(ctf)

            self.add_variance(ind, variance, exp_ref, exp_spectrum, ref_spectrum, ctf_spectrum)
            self.update_scale_correction()
            #print(self.group_scales)
        return variance
        #grid_variance = self.ctf_grid.shell_to_grid(self.group_variances[group_ids, :])
        #plt.figure()
        #plt.imshow(np.log(grid_variance[0].cpu().numpy()))
        #print(grid_variance[0])

    def reset_group_counts(self):
        self.group_counts = torch.zeros(self.max_group_no + 1,)

    def update_scale_correction(self, eps=1e-5):
        self.group_scales = self.group_exp_ref_spectrum.sum(-1) / (self.group_ref_spectrum.sum(-1) + eps)
        total_ = (self.group_scales != 0).sum()
        assert total_ > 0
        avg_scale = self.group_scales.sum()/total_
        self.group_scales /= (avg_scale + eps)

    def plot_variance(self, ind):
        g_id = self.group[ind]
        x = np.arange(self.max_r.detach().cpu().numpy())
        f, axes = plt.subplots(1, 1)
        #plt.figure()
        #print(self.group_ctf_spectrum[g_id].detach().cpu().numpy().shape)
        b, b_profile = self.fit_b_factor(g_id, 80)
        #axes[0].plot(x, np.log(self.group_exp_ref_spectrum[g_id][:self.max_r].abs().detach().cpu().numpy() + 1e-5)
        #              - np.log(self.group_ref_spectrum[g_id][:self.max_r].detach().cpu().numpy() + 1e-5))
        #axes[0].plot(x[:b_profile.shape[0]], b_profile.detach().cpu().numpy())

        #axes[0].plot(x, np.sqrt(self.group_ctf_spectrum[g_id][:self.max_r].detach().cpu().numpy() + 1e-5), '-.')
        axes.plot(x, np.log((self.group_exp_spectrum[g_id]/self.group_counts[g_id])[:self.max_r].detach().cpu().numpy() + 1e-5), '-.')
        axes.plot(x, np.log((self.group_ref_spectrum[g_id]/self.group_counts[g_id])[:self.max_r].detach().cpu().numpy() + 1e-5), '--')
        #axes[0].plot(x, np.log(self.group_ref_spectrum[g_id].detach().cpu().numpy()/self.group_exp_spectrum[g_id].detach().cpu().numpy()), '-o')
        #axes[0].imshow(self.ctf_grid.shells_index.detach().cpu().numpy())
        #axes[0].plot(x, self.ctf_grid.shells_count.detach().cpu().numpy())
        axes.plot(x, np.log(((self.group_exp_ref_spectrum[g_id]/self.group_counts[g_id])[:self.max_r].detach().cpu().numpy() + 1e-5)), '-')
        axes.plot(x, np.log(self.group_variances[g_id][:self.max_r].detach().cpu().numpy() + 1e-5), '-')
        print(self.group_variances[g_id][0].cpu().numpy())
        #w_filt = self.get_wiener_filter(ind)
        #axes.plot(x, np.log(w_filt[:self.max_r].detach().cpu().numpy() + 1e-5), '-.')
        #plt.show()

    @classmethod
    def load(cls, group, infile, device, D, vol_size, optimize_b):
        '''
        Return an instance of GroupStat

        Inputs:
            infile (str or list):   One or two files, with format options of:
                                    single file with pose pickle
                                    two files with rot and trans pickle
                                    single file with rot pickle
            Nimg:               Number of particles
            D:                  Box size (pixels)
        '''
        # load pickle
        group_stats = torch.load(infile)
        print(group_stats['group_b_factors'])

        return cls(group, device, D=D, vol_size=vol_size, group_stats=group_stats, optimize_b=optimize_b)

    def save(self, out_pkl):
        group_stats = {}
        group_stats['group_variances'] = self.group_variances
        group_stats['group_exp_ref_spectrum'] = self.group_exp_ref_spectrum
        group_stats['group_exp_spectrum'] = self.group_exp_spectrum
        group_stats['group_ref_spectrum'] = self.group_ref_spectrum
        group_stats['group_ctf_spectrum'] = self.group_ctf_spectrum
        group_stats['group_counts'] = self.group_counts
        group_stats['group_scales'] = self.group_scales
        if self.optimize_b:
            group_stats['group_b_factors'] = self.group_b_factors
        torch.save(group_stats, out_pkl)

