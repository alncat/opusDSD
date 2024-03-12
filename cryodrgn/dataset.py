import numpy as np
import torch
from torch.utils import data
import os
import multiprocessing as mp
from multiprocessing import Pool

from . import fft
from . import mrc
from . import utils
from . import starfile

log = utils.log

def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False, return_header=False):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
            else: raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, header = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    if not return_header:
        return particles
    else:
        return particles, header


class LazyMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 window=True, datadir=None, relion31=False, window_r=0.85, in_mem=True):
        #assert not keepreal, 'Not implemented error'
        particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
        N = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        self.real_data = real_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        #self.window = window_mask(ny, window_r, .99) if window else None
        self.window = window_cos_mask(ny, window_r, .95) if window else None
        if in_mem:
            log('Reading all images into memory!')
            self.particles = np.asarray([self.particles[i].get() for i in range(0, self.N)])
        self.in_mem = in_mem

    def estimate_normalization(self, n=1000):
        n = min(n,self.N)
        if self.real_data:
            imgs = np.nan_to_num(np.asarray([self.particles[i].get() for i in range(0,self.N, self.N//n)]))
        else:
            imgs = np.asarray([fft.ht2_center(self.particles[i].get()) for i in range(0,self.N, self.N//n)])
        if self.invert_data: imgs *= -1
        if not self.real_data:
            imgs = fft.symmetrize_ht(imgs)
        log('first image: {}'.format(imgs[0]))
        norm = [np.mean(imgs), np.std(imgs)]
        if self.real_data:
            log('Image Mean, Std are {} +/- {}'.format(*norm))
        else:
            log('Normalizing HT by {} +/- {}'.format(*norm))
        norm[0] = 0
        return norm

    def get(self, i):
        if self.in_mem:
            return np.nan_to_num(self.particles[i])
        img = np.nan_to_num(self.particles[i].get())
        if self.window is not None:
            img *= self.window
        if self.real_data:
            return img
        img = fft.ht2_center(img).astype(np.float32)
        if self.invert_data: img *= -1
        img = fft.symmetrize_ht(img)
        img = (img - self.norm[0])/self.norm[1]
        return img

    def get_batch(self, batch):
        imgs = []
        for i in range(len(batch)):
            imgs.append(self.get(batch[i]))
        return np.concatenate(imgs, axis=0)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get(index), index

class ClassSplitBatchSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind, split):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors
        #filter poses_ind not in split
        poses_ind_new = []
        for x in self.poses_ind:
            poses_ind_new.append(x[np.isin(x.numpy(), split.numpy())])
        self.poses_ind = poses_ind_new
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]

        self.num_samples = sum(self.ns)
        print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(n) for n in self.ns]
        #print("rand_perms: ", rand_perms)
        print("ns: ", self.ns)
        print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if found:
                start = current_ind[rand_pose]
                current_ind[rand_pose] += self.batch_size
                current_num_samples += self.batch_size

                sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
                #indexing poses_ind
                yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

class ClassBatchSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]

        self.num_samples = sum(self.ns)
        print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(n) for n in self.ns]
        #print("rand_perms: ", rand_perms)
        print("ns: ", self.ns)
        print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if found:
                start = current_ind[rand_pose]
                current_ind[rand_pose] += self.batch_size
                current_num_samples += self.batch_size

                sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
                #indexing poses_ind
                yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask

def window_cos_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1., np.maximum(0.0, (r-in_rad)/(out_rad - in_rad)))
    mask = 0.5 + 0.5*np.cos(mask*np.pi)
    return mask

class VolData(data.Dataset):
    '''
    Class representing an .mrcs stack file
    '''
    def __init__(self, mrcfile, norm=None, invert_data=False, datadir=None, relion31=False, max_threads=16, window_r=0.85):
        particles, header = load_particles(mrcfile, False, datadir=datadir, relion31=relion31, return_header=True)
        N, ny, nx = particles.shape
        assert N == ny == nx, "Images must be cubic"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {}x{}x{} images with Apix {}'.format(N, ny, nx, header.get_apix()))

        if invert_data: particles *= -1

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        #particles = (particles - norm[0])/norm[1]
        #log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.volume = torch.from_numpy(self.particles)
        self.N = N
        self.D = particles.shape[1] # ny
        self.norm = norm
        self.Apix = header.get_apix()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self):
        return self.volume

    def center_of_mass(self,):
        mass = ((self.volume > 0)*self.volume).sum()
        x_idx = torch.linspace(0, self.N-1, self.N) - self.N/2 #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1)
        vol = ((self.volume > 0).float()*self.volume).unsqueeze(-1)
        center = vol*grid
        center = center.sum(dim=(0,1,2))
        assert mass.item() > 0
        center /= mass
        center = torch.where(center > 0, (center + 0.5).int(), (center - 0.5).int()).float()
        centered = (grid - center)*vol
        radius = (centered).pow(2)
        r = torch.sqrt(radius.sum(dim=(0,1,2))/mass)
        #principal axes
        matrix = -centered.unsqueeze(-1) * centered.unsqueeze(-2)
        radius_sum = torch.eye(3) * (radius.sum(dim=-1, keepdim=True).unsqueeze(-1))
        matrix = ((matrix+radius_sum)*vol.unsqueeze(-1)).sum(dim=(0, 1, 2))
        eigvals, eigvecs = np.linalg.eig(matrix.numpy())
        indices = np.argsort(eigvals)
        #print(matrix, eigvals[indices])
        eigvecs = torch.from_numpy(eigvecs[:, indices].T) # eigvecs[0] is the first eigen vector with largest eigenvalues
        #print(eigvecs @ eigvecs[2])

        return center, r, eigvecs


class MRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file
    '''
    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, relion31=False, max_threads=16, window_r=0.85):
        if keepreal:
            raise NotImplementedError
        if ind is not None:
            particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
            particles = np.array([particles[i].get() for i in ind])
        else:
            particles = load_particles(mrcfile, False, datadir=datadir, relion31=relion31)
        N, ny, nx = particles.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))

        # Real space window
        if window:
            log(f'Windowing images with radius {window_r}')
            particles *= window_mask(ny, window_r, .99)

        # compute HT
        log('Computing FFT')
        max_threads = min(max_threads, mp.cpu_count())
        if max_threads > 1:
            log(f'Spawning {max_threads} processes')
            with Pool(max_threads) as p:
                particles = np.asarray(p.map(fft.ht2_center, particles), dtype=np.float32)
        else:
            particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32)
            log('Converted to FFT')

        if invert_data: particles *= -1

        # symmetrize HT
        log('Symmetrizing image data')
        particles = fft.symmetrize_ht(particles)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.N = N
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            log('Normalized real space images by {}'.format(particles_real.std()))
            self.particles_real /= particles_real.std()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class PreprocessedMRCData(data.Dataset):
    '''
    '''
    def __init__(self, mrcfile, norm=None, ind=None):
        particles = load_particles(mrcfile, False)
        if ind is not None:
            particles = particles[ind]
        log(f'Loaded {len(particles)} {particles.shape[1]}x{particles.shape[1]} images')
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))
        self.particles = particles
        self.N = len(particles)
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class TiltMRCData(data.Dataset):
    '''
    Class representing an .mrcs tilt series pair
    '''
    def __init__(self, mrcfile, mrcfile_tilt, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, window_r=0.85):
        if ind is not None:
            particles_real = load_particles(mrcfile, True, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, True, datadir)
            particles_real = np.array([particles_real[i].get() for i in ind], dtype=np.float32)
            particles_tilt_real = np.array([particles_tilt_real[i].get() for i in ind], dtype=np.float32)
        else:
            particles_real = load_particles(mrcfile, False, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, False, datadir)

        N, ny, nx = particles_real.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        assert particles_tilt_real.shape == (N, ny, nx), "Tilt series pair must have same dimensions as untilted particles"
        log('Loaded {} {}x{} tilt pair images'.format(N, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles_real *= m
            particles_tilt_real *= m

        # compute HT
        particles = np.asarray([fft.ht2_center(img) for img in particles_real]).astype(np.float32)
        particles_tilt = np.asarray([fft.ht2_center(img) for img in particles_tilt_real]).astype(np.float32)
        if invert_data:
            particles *= -1
            particles_tilt *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles)
        particles_tilt = fft.symmetrize_ht(particles_tilt)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        particles_tilt = (particles_tilt - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.particles_tilt = particles_tilt
        self.norm = norm
        self.N = N
        self.D = particles.shape[1]
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            self.particles_tilt_real = particles_tilt_real

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], self.particles_tilt[index], index

    def get(self, index):
        return self.particles[index], self.particles_tilt[index]

# TODO: LazyTilt
