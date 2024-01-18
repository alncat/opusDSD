import torch
import torch.nn as nn
import numpy as np
import pickle
import healpy as hp
from sklearn.cluster import KMeans

from . import lie_tools
from . import utils
log = utils.log

class PoseTracker(nn.Module):
    def __init__(self, rots_np, trans_np=None, D=None, emb_type=None, deform=False, deform_emb_size=2, eulers_np=None,
                 latents=None, batch_size=None, body_eulers_np=None, body_trans_np=None):
        super(PoseTracker, self).__init__()
        rots = torch.tensor(rots_np).float()
        trans = torch.tensor(trans_np).float() if trans_np is not None else None
        self.eulers = torch.tensor(eulers_np).float() if eulers_np is not None else None
        self.body_eulers = torch.tensor(body_eulers_np).float() if body_eulers_np is not None else None
        self.body_trans = torch.tensor(body_trans_np).float() if body_trans_np is not None else None
        self.rots = rots
        self.trans = trans
        self.use_trans = trans_np is not None
        self.D = D
        self.emb_type = emb_type
        self.deform = deform
        self.deform_emb = None
        if emb_type is None:
            pass
        else:
            if self.use_trans:
                trans_emb = nn.Embedding(trans.shape[0], 2, sparse=True)
                trans_emb.weight.data.copy_(trans)
            if emb_type == 's2s2':
                rots_emb = nn.Embedding(rots.shape[0], 6, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_s2s2(rots))
            elif emb_type == 'quat':
                rots_emb = nn.Embedding(rots.shape[0], 4, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_quaternions(rots))
            else:
                raise RuntimeError('Embedding type {} not recognized'.format(emb_type))
            self.rots_emb = rots_emb
            self.trans_emb = trans_emb if self.use_trans else None
        if self.deform:
            self.deform_emb_size = deform_emb_size
            #deform_emb = torch.zeros(rots.shape[0], deform_emb_size)
            #if encoding is not None:
            #    emb_data = encoding.repeat(rots.shape[0], 1)
            #    print(emb_data.shape)
            #else:
            #    emb_data = torch.randn(rots.shape[0], deform_emb_size)
            #deform_emb.weight.data.copy_(emb_data)
            #self.deform_emb = deform_emb
            # convert euler to hopf
            self.hopfs = lie_tools.euler_to_hopf(self.eulers)
            new_eulers = lie_tools.hopf_to_euler(self.hopfs)
            #print(self.eulers[30, :], self.hopfs.numpy()[30,:], np.where(np.isnan(new_eulers.numpy())))
            print("euler difference: ", torch.sum((self.eulers - new_eulers).abs())/self.hopfs.shape[0], self.hopfs.shape[0])
            print("max difference: ", torch.max((self.eulers - new_eulers).abs(), dim=0))
            # convert poses to healpix indices
            print(eulers_np[:5, :])
            print(self.hopfs[:5, :])

            #reset euler using hopf
            self.eulers = self.hopfs
            eulers_np = self.hopfs.cpu().numpy()
            self.decoder_eulers = None
            if self.decoder_eulers is not None:
                #convert euler to hopf
                self.decoder_eulers = lie_tools.euler_to_hopf(self.decoder_eulers)
                print("hopf difference between encoder and decoder: ", torch.sum((self.eulers - self.decoder_eulers).abs())/self.hopfs.shape[0], self.hopfs.shape[0])

            euler0 = eulers_np[:, 0]*np.pi/180 #(-180, 180)
            euler1 = eulers_np[:, 1]*np.pi/180 #(0, 180)
            self.hp_order = 2
            euler_pixs = hp.ang2pix(self.hp_order, euler1, euler0, nest=True)
            num_pixs   = self.hp_order**2*12
            self.poses_ind = [[] for i in range(num_pixs)]
            for i in range(len(euler_pixs)):
                assert euler_pixs[i] < num_pixs
                self.poses_ind[euler_pixs[i]].append(i)
            self.poses_ind = [torch.tensor(x) for x in self.poses_ind]
            self.euler_groups = euler_pixs
            if latents is not None:
                #self.mu = latents
                self.mu = latents["mu"]
                self.nearest_poses = latents["nn"]
                if "multi_mu" in latents:
                    self.multi_mu = latents["multi_mu"]
                else:
                    self.multi_mu = None #torch.randn(rots.shape[0], 4)
            else:
                self.mu = torch.randn(rots.shape[0], self.deform_emb_size)
                self.nearest_poses = [np.array([], dtype=np.int64) for i in range(len(euler_pixs))]
                self.multi_mu = None #torch.randn(rots.shape[0], 4)
            self.batch_size = batch_size
            print("nn: ", len(self.nearest_poses), "batch_size: ", self.batch_size)
            self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]
            self.total_ns = sum(self.ns)
            print(self.ns)
            self.valid_poses = []
            for i in range(num_pixs):
                if self.ns[i] > 0:
                    self.valid_poses.append(i)
            #print("poses_ind: ", self.poses_ind)
            print(len(euler_pixs), len(eulers_np), self.valid_poses)

    def filter_poses_ind(self, split):
        poses_ind_new = []
        for x in self.poses_ind:
            poses_ind_new.append(x[np.isin(x.numpy(), split.numpy())])
        self.poses_ind = poses_ind_new
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]

    def sample_full_neighbors(self, euler, inds, num_pose=8):
        cur_idx = self.euler_groups[inds[0]]
        euler0 = euler[0, 0]*np.pi/180
        euler1 = euler[0, 1]*np.pi/180
        cur_idx_ = hp.ang2pix(self.hp_order, euler1, euler0, nest=True)
        #assert cur_idx == cur_idx_
        pose_sample = list(self.valid_poses)
        # remove current pose
        pose_sample.remove(cur_idx)
        num_pose = min(len(pose_sample), num_pose)
        perm = np.random.choice(pose_sample, size=num_pose, replace=False)
        total = sum([self.ns[i] for i in perm])
        sample_idices = []
        sample_mus = []
        #print(cur_idx, perm)
        total_samples = 256*50
        for i in range(len(perm)):
            #pose_idx = pose_sample[i] #
            pose_idx = perm[i]
            # sample from selected pose
            samples = np.random.choice(self.ns[pose_idx],
                    size=min(int(self.ns[pose_idx]/total*total_samples), self.ns[pose_idx]), replace=False)
            #print(samples)
            idx_ = self.poses_ind[pose_idx][samples]
            sample_idices.append(idx_)
            sample_mus.append(torch.cat([self.mu[idx_,:], self.multi_mu[idx_,:]], dim=-1))
        #print(total)
        sample_idices = np.concatenate(sample_idices, axis=0)
        sample_mus = torch.cat(sample_mus, dim=0)

        # compare with current nearest neighbors
        mus = []
        top_indices = []
        top_mus = []
        neg_mus = []
        num_samples = 128
        num_mu_samples = len(sample_mus)
        num_samples = min(num_samples, num_mu_samples//4)
        for i in range(len(inds)):
            global_i  = inds[i]
            n_i = sample_idices
            mu_ = sample_mus[..., :]
            mu_i = torch.cat([self.mu[global_i, :], self.multi_mu[global_i, :]], dim=-1)

            diff = (mu_i - mu_).pow(2).sum(-1)
            top = torch.topk(diff, k=num_samples, largest=False, sorted=True)
            # gather output
            top_mu_ = mu_[top.indices, :]
            mus.append(top_mu_)

            neg = torch.topk(diff, k=num_samples*4, largest=True, sorted=True)

            # uniform sample
            uni_sample = np.random.choice(len(diff), size=num_samples*4, replace=False)
            uni_mu = mu_[uni_sample, :]

            # mix samples
            neg_mu = mu_[neg.indices, :]
            neg_mu = 0.8*neg_mu + 0.2*uni_mu

            neg_mus.append(neg_mu)
            #cur_inds = np.delete(inds, [i])
            #cur_mus = self.mu[cur_inds, :]
            #diff = (mu_i - cur_mus).pow(2).sum(-1)
            #top = torch.topk(diff, k=3, largest=False, sorted=True)
            top_indices.append(torch.tensor(n_i[top.indices[0]]))
            #top_indices.append(torch.tensor(n_i[top.indices[:6]]))
            #top_mus.append(self.mu[cur_inds[top.indices], :])

        mus = torch.stack(mus, dim=0)
        neg_mus = torch.stack(neg_mus, dim=0)
        top_indices = torch.stack(top_indices, dim=0).view(-1) #(B*k)
        top_mus = None #torch.stack(top_mus, dim=0)
        #print(mus.shape, top_indices.shape, top_mus.shape)
        return mus, top_indices, top_mus, neg_mus

    def sample_neighbors(self, euler, inds, num_pose=8):
        cur_idx = self.euler_groups[inds[0]]
        euler0 = euler[0, 0]*np.pi/180
        euler1 = euler[0, 1]*np.pi/180
        cur_idx_ = hp.ang2pix(self.hp_order, euler1, euler0, nest=True)
        #assert cur_idx == cur_idx_
        pose_sample = list(self.valid_poses)
        # remove current pose
        pose_sample.remove(cur_idx)
        num_pose = min(len(pose_sample), num_pose)
        perm = np.random.choice(pose_sample, size=num_pose, replace=False)
        total = sum([self.ns[i] for i in perm])
        sample_idices = []
        sample_mus = []
        #print(cur_idx, perm)
        total_samples = 256*50
        for i in range(len(perm)):
            #pose_idx = pose_sample[i] #
            pose_idx = perm[i]
            # sample from selected pose
            samples = np.random.choice(self.ns[pose_idx],
                    size=min(int(self.ns[pose_idx]/total*total_samples), self.ns[pose_idx]), replace=False)
            #print(samples)
            idx_ = self.poses_ind[pose_idx][samples]
            sample_idices.append(idx_)
            sample_mus.append(self.mu[idx_,:])
        #print(total)
        sample_idices = np.concatenate(sample_idices, axis=0)
        sample_mus = torch.cat(sample_mus, dim=0)

        # compare with current nearest neighbors
        mus = []
        top_indices = []
        top_mus = []
        neg_mus = []
        num_samples = 128
        num_mu_samples = len(sample_mus)
        num_samples = min(num_samples, num_mu_samples//4)
        for i in range(len(inds)):
            global_i  = inds[i]
            n_i = sample_idices
            mu_ = sample_mus[..., :]
            mu_i = self.mu[global_i, :]

            diff = (mu_i - mu_).pow(2).sum(-1)
            top = torch.topk(diff, k=num_samples, largest=False, sorted=True)
            # gather output
            top_mu_ = mu_[top.indices, :]
            mus.append(top_mu_)

            neg = torch.topk(diff, k=num_samples*4, largest=True, sorted=True)

            # uniform sample
            uni_sample = np.random.choice(len(diff), size=num_samples*4, replace=False)
            uni_mu = mu_[uni_sample, :]

            # mix samples
            neg_mu = mu_[neg.indices, :]
            neg_mu = 0.8*neg_mu + 0.2*uni_mu

            neg_mus.append(neg_mu)
            #cur_inds = np.delete(inds, [i])
            #cur_mus = self.mu[cur_inds, :]
            #diff = (mu_i - cur_mus).pow(2).sum(-1)
            #top = torch.topk(diff, k=3, largest=False, sorted=True)
            top_indices.append(torch.tensor(n_i[top.indices[0]]))
            #top_indices.append(torch.tensor(n_i[top.indices[:6]]))
            #top_mus.append(self.mu[cur_inds[top.indices], :])

        mus = torch.stack(mus, dim=0)
        neg_mus = torch.stack(neg_mus, dim=0)
        top_indices = torch.stack(top_indices, dim=0).view(-1) #(B*k)
        top_mus = None #torch.stack(top_mus, dim=0)
        #print(mus.shape, top_indices.shape, top_mus.shape)
        return mus, top_indices, top_mus, neg_mus

    def set_emb(self, encodings, ind, mu=0.7):
        #when the encoding dim is larger, we knew the multi-body refinement is on
        if encodings.shape[-1] > self.deform_emb_size:
            if self.multi_mu is None:
                #initialize multi mu
                self.multi_mu = torch.randn(len(self.mu), encodings.shape[-1] - self.deform_emb_size)
                log(f"intializing multi_mu of {len(self.mu)}, {self.multi_mu.shape[-1]}")
            if self.multi_mu.shape[-1] != encodings.shape[-1] - self.deform_emb_size:
                self.multi_mu = torch.randn(len(self.mu), encodings.shape[-1] - self.deform_emb_size)
                log(f"reintializing multi_mu of {len(self.mu)}, {self.multi_mu.shape[-1]}")
            self.mu[ind] = self.mu[ind]*mu + (1-mu)*encodings[:, :self.deform_emb_size].detach().cpu()
            self.multi_mu[ind] = self.multi_mu[ind]*mu + (1-mu)*encodings[:, self.deform_emb_size:].detach().cpu()
        else:
            self.mu[ind] = self.mu[ind]*mu + (1-mu)*encodings.detach().cpu()

    def save_emb(self, filename):
        torch.save({"mu": self.mu, "nn": self.nearest_poses, "multi_mu": self.multi_mu}, filename)

    @classmethod
    def load(cls, infile, Nimg, D, emb_type=None, ind=None, deform=False, deform_emb_size=2, latents=None, batch_size=None, decoder_infile=None):
        '''
        Return an instance of PoseTracker

        Inputs:
            infile (str or list):   One or two files, with format options of:
                                    single file with pose pickle
                                    two files with rot and trans pickle
                                    single file with rot pickle
            Nimg:               Number of particles
            D:                  Box size (pixels)
            emb_type:           SO(3) embedding type if refining poses
            ind:                Index array if poses are being filtered
        '''
        # load pickle
        if type(infile) is str: infile = [infile]
        assert len(infile) in (1,2)
        if len(infile) == 2: # rotation pickle, translation pickle
            poses = (utils.load_pkl(infile[0]), utils.load_pkl(infile[1]))
        else: # rotation pickle or poses pickle
            poses = utils.load_pkl(infile[0])
            if decoder_infile is not None:
                decoder_poses = utils.load_pkl(decoder_infile)
            else:
                decoder_poses = None
            if type(poses) != tuple: poses = (poses,)

        # rotations
        rots = poses[0]
        if ind is not None:
            if len(rots) > Nimg: # HACK
                rots = rots[ind]
        assert rots.shape == (Nimg,3,3), f"Input rotations have shape {rots.shape} but expected ({Nimg},3,3)"

        body_eulers, body_trans = None, None

        # translations if they exist
        if len(poses) == 2:
            trans = poses[1]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    trans = trans[ind]
            assert trans.shape == (Nimg,2), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
            assert np.all(trans <= 1), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D # convert from fraction to pixels
        elif len(poses) == 3:
            trans = poses[1]
            decoder_trans = decoder_poses[1] if decoder_poses is not None else None
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    trans = trans[ind]
            assert trans.shape == (Nimg,2) or trans.shape == (Nimg, 3), f"Input translations have shape {trans.shape} but expected ({Nimg},2/3)"
            assert np.all(trans <= 1), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D # convert from fraction to pixels
            log("loaded eulers")
            eulers = poses[2]
            decoder_eulers = decoder_poses[2] if decoder_poses is not None else None
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    eulers = eulers[ind]
            assert eulers.shape == (Nimg,3), f"Input eulers have shape {trans.shape} but expected ({Nimg},3)"
        elif len(poses) > 3:
            trans = poses[1]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    trans = trans[ind]
            assert trans.shape == (Nimg,2), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
            assert np.all(trans <= 1), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D # convert from fraction to pixels
            log("loaded eulers")
            eulers = poses[2]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    eulers = eulers[ind]
            assert eulers.shape == (Nimg,3), f"Input eulers have shape {eulers.shape} but expected ({Nimg},3)"
            body_eulers = poses[3]
            body_trans = poses[4]
            assert body_eulers.shape[0] == Nimg and body_eulers.shape[2] == 3, f"Input eulers have shape {body_eulers.shape} but expected ({Nimg},3)"
            assert body_trans.shape[0] == Nimg and body_trans.shape[2] == 2 and body_trans.shape[1] == body_eulers.shape[1], \
                                f"Input translations have shape {body_trans.shape} but expected ({Nimg},2)"

        else:
            log('WARNING: No translations provided')
            trans = None
            eulers = None

        if latents is not None:
            latents = torch.load(latents)
        return cls(rots, trans, D, emb_type, deform, deform_emb_size, eulers, latents, batch_size, body_eulers_np=body_eulers, body_trans_np=body_trans)

    def save_decoder_pose(self, out_pkl):
        r = self.rots.cpu().numpy()
        t = self.decoder_trans.cpu().numpy()
        t = t/self.D # convert from pixels to extent
        # convert from hopf back to euler
        new_eulers = lie_tools.hopf_to_euler(self.decoder_eulers)
        e = new_eulers.cpu().numpy()
        poses = (r,t,e)
        pickle.dump(poses, open(out_pkl,'wb'))

    def save(self, out_pkl):
        if self.emb_type == 'quat':
            r = lie_tools.quaternions_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        elif self.emb_type == 's2s2':
            r = lie_tools.s2s2_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        else:
            r = self.rots.cpu().numpy()

        if self.use_trans:
            if self.emb_type is None:
                t = self.trans.cpu().numpy()
            else:
                t = self.trans_emb.weight.data.cpu().numpy()
            t = t/self.D # convert from pixels to extent
            if self.eulers is not None:
                # convert from hopf back to euler
                new_eulers = lie_tools.hopf_to_euler(self.eulers)
                #e = self.eulers.cpu().numpy()
                e = new_eulers.cpu().numpy()
                poses = (r,t,e)
            else:
                poses = (r,t)
        else:
            poses = (r,)

        pickle.dump(poses, open(out_pkl,'wb'))

    def get_euler(self, ind):
        if self.emb_type is None:
            euler = self.eulers[ind]
            return euler

    def set_euler(self, euler, ind):
        self.eulers[ind] = euler

    def set_body_euler(self, euler, trans, ind):
        self.eulers[ind] = euler
        self.trans[ind] = trans

    def get_body_pose(self, ind):
        if self.body_eulers is not None:
            euler = self.body_eulers[ind]
            trans = self.body_trans[ind]
            return euler, trans
        else:
            return None, None

    def get_pose(self, ind):
        if self.emb_type is None:
            rot = self.rots[ind]
            tran = self.trans[ind] if self.use_trans else None
        else:
            if self.emb_type == 's2s2':
                rot = lie_tools.s2s2_to_SO3(self.rots_emb(ind))
            elif self.emb_type == 'quat':
                rot = lie_tools.quaternions_to_SO3(self.rots_emb(ind))
            else:
                raise RuntimeError # should not reach here
            tran = self.trans_emb(ind) if self.use_trans else None
        #if self.deform:
        #    defo = self.deform_emb(ind)
        #    return rot, tran, defo
        return rot, tran
