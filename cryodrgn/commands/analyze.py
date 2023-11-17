'''
Visualize latent space and generate volumes
'''

import argparse
import numpy as np
import math
import sys, os
import pickle
import shutil
import healpy as hp
from datetime import datetime as dt
import scipy

import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import cryodrgn
from cryodrgn import analysis
from cryodrgn import utils
from cryodrgn.pose import PoseTracker

log = utils.log

def add_args(parser):
    parser.add_argument('workdir', type=os.path.abspath, help='Directory with cryoDRGN results')
    parser.add_argument('epoch', type=int, help='Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)')
    parser.add_argument('--device', type=int, help='Optionally specify CUDA device')
    parser.add_argument('-o','--outdir', help='Output directory for analysis results (default: [workdir]/analyze.[epoch])')
    parser.add_argument('--skip-vol', action='store_true', help='Skip generation of volumes')
    parser.add_argument('--skip-umap', action='store_true', help='Skip running UMAP')
    parser.add_argument('--vanilla', action='store_true', help='Skip running UMAP')
    parser.add_argument('--D', type=int, help='Skip running UMAP')
    parser.add_argument('--pose', help='directory for analysis results (default: [workdir]/analyze.[epoch])')

    group = parser.add_argument_group('Extra arguments for volume generation')
    group.add_argument('--Apix', type=float, default=1, help='Pixel size to add to .mrc header (default: %(default)s A/pix)')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('-d','--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--pc', type=int, default=2, help='Number of principal component traversals to generate (default: %(default)s)')
    group.add_argument('--ksample', type=int, default=20, help='Number of kmeans samples to generate (default: %(default)s)')
    group.add_argument('--psample', type=int, default=10, help='Number of pc samples to generate (default: %(default)s)')
    return parser

def analyze_z1(z, outdir, vg):
    '''Plotting and volume generation for 1D z'''
    assert z.shape[1] == 1
    z = z.reshape(-1)
    N = len(z)

    plt.figure(1)
    plt.scatter(np.arange(N), z, alpha=.1, s=2)
    plt.xlabel('particle')
    plt.ylabel('z')
    plt.savefig(f'{outdir}/z.png')

    plt.figure(2)
    sns.distplot(z)
    plt.xlabel('z')
    plt.savefig(f'{outdir}/z_hist.png')

    ztraj = np.linspace(*np.percentile(z,(5,95)), 10) # or np.percentile(z, np.linspace(5,95,10)) ?
    vg.gen_volumes(outdir, ztraj)

def analyze_zN(z, outdir, vg, groups, skip_umap=False, num_pcs=2, num_ksamples=20, num_pc_samples=10):
    zdim = z.shape[1]

    # Principal component analysis
    print(z[:5, :])
    log('Perfoming principal component analysis...')
    pc, pca = analysis.run_pca(z)
    log('Generating volumes...')
    for i in range(num_pcs):
        start, end = np.percentile(pc[:,i],(5,95))
        log(f'traversing pc {i} from {start} to {end}')
        z_pc = analysis.get_pc_traj(pca, z.shape[1], num_pc_samples, i+1, start, end)
        if not os.path.exists(f'{outdir}/pc{i+1}'):
            os.mkdir(f'{outdir}/pc{i+1}')
        vg.gen_volumes(f'{outdir}/pc{i+1}', z_pc)
        np.savetxt(f'{outdir}/pc{i+1}/z_pc.txt', z_pc)

    # kmeans clustering
    log('K-means clustering...')
    K = num_ksamples
    kmeans_labels, centers = analysis.cluster_kmeans(z, K)
    _, centers_ind = analysis.get_nearest_point(z, centers)
    if not os.path.exists(f'{outdir}/kmeans{K}'):
        os.mkdir(f'{outdir}/kmeans{K}')
    utils.save_pkl(kmeans_labels, f'{outdir}/kmeans{K}/labels.pkl')
    utils.save_pkl(centers, f'{outdir}/kmeans{K}/centers.pkl')
    np.savetxt(f'{outdir}/kmeans{K}/centers.txt', centers)
    np.savetxt(f'{outdir}/kmeans{K}/centers_ind.txt', centers_ind, fmt='%d')
    log('Generating volumes...')
    vg.gen_volumes(f'{outdir}/kmeans{K}', centers)
    print(np.bincount(kmeans_labels))

    # UMAP -- slow step
    if zdim > 2:
        log('Running UMAP...')
        umap_file = f'{outdir}/umap.pkl'
        if os.path.exists(umap_file) and skip_umap:
            umap_emb = utils.load_pkl(umap_file)
            skip_umap = False
        else:
            umap_emb = analysis.run_umap(z)
            utils.save_pkl(umap_emb, umap_file)
            skip_umap = False

    #log('Running TSNE...')
    #tsne_file = f'{outdir}/tsne.pkl'
    #if os.path.exists(tsne_file):
    #    tsne_emb = utils.load_pkl(tsne_file)
    #else:
    #    tsne_emb = analysis.run_tsne(z)
    #    utils.save_pkl(tsne_emb, tsne_file)

    # Make some plots
    log('Generating plots...')
    plt.figure(1)
    g = sns.jointplot(x=pc[:,0], y=pc[:,1], alpha=.1, s=2)
    g.set_axis_labels('PC1','PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca.png')

    plt.figure(2)
    g = sns.jointplot(x=pc[:,0], y=pc[:,1], kind='hex')
    g.set_axis_labels('PC1','PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca_hexbin.png')

    if zdim > 2 and not skip_umap:
        xmax = np.max(umap_emb[:, 0])
        xmin = np.min(umap_emb[:, 0])
        ymax = np.max(umap_emb[:, 1])
        ymin = np.min(umap_emb[:, 1])
        interval = max(xmax-xmin, ymax-ymin)
        interval = math.ceil(interval)
        #interval = interval + (interval%2)
        log(f"using interval {interval}")
        plt.figure(3)
        g = sns.jointplot(x=umap_emb[:,0], y=umap_emb[:,1], hue=groups, palette="icefire", s=1.5, alpha=.2, xlim=(xmin, xmin+interval), ylim=(ymin, ymin+interval))
        g.set_axis_labels('UMAP1','UMAP2')
        #plt.tight_layout()
        plt.savefig(f'{outdir}/umap.png')

        plt.figure(4)
        g = sns.jointplot(x=umap_emb[:,0], y=umap_emb[:,1], kind='hex')
        g.set_axis_labels('UMAP1','UMAP2')
        #plt.tight_layout()
        plt.savefig(f'{outdir}/umap_hexbin.png')

        #plt.figure(5)
        #g = sns.jointplot(x=tsne_emb[:,0], y=tsne_emb[:,1], hue=groups, palette="vlag", s=2)
        #g.set_axis_labels('TSNE1','TSNE2')
        #plt.tight_layout()
        #plt.savefig(f'{outdir}/tsne.png')

    analysis.scatter_annotate(pc[:,0], pc[:,1], centers_ind=centers_ind, annotate=True)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'{outdir}/kmeans{K}/z_pca.png')

    g = analysis.scatter_annotate_hex(pc[:,0], pc[:,1], centers_ind=centers_ind, annotate=True)
    g.set_axis_labels('PC1','PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/kmeans{K}/z_pca_hex.png')

    if zdim > 2 and not skip_umap:
        fig, ax = plt.subplots()
        ax.scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=.1, s=.5, rasterized=True,)
        ax.set_xlim(xmin, xmin+interval)
        ax.set_ylim(ymin, ymin+interval)
        plt.gca().set_aspect('equal')
        ax.axis('off')
        plt.savefig(f'{outdir}/kmeans{K}/scatter.png', dpi=180)

        analysis.scatter_annotate(umap_emb[:,0], umap_emb[:,1], centers_ind=centers_ind, annotate=True,
                                  xlim=(xmin, xmin+interval), ylim=(ymin, ymin+interval),
                                  alpha=.1, s=.5)
        plt.xlabel('UMAP1', fontsize=14, weight='bold')
        plt.ylabel('UMAP2', fontsize=14, weight='bold')
        plt.savefig(f'{outdir}/kmeans{K}/centers.svg')

        g, _ = analysis.scatter_annotate(umap_emb[:,0], umap_emb[:,1], centers_ind=centers_ind, annotate=True,
                                      xlim=(xmin, xmin+interval), ylim=(ymin, ymin+interval),
                                      alpha=.1, s=.5, plot_scatter=True)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        #plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{K}/umap.png')

    for i in range(num_pcs):
        if zdim > 2 and not skip_umap:
            analysis.scatter_color(umap_emb[:,0], umap_emb[:,1], pc[:,i], label=f'PC{i+1}')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.tight_layout()

            plt.savefig(f'{outdir}/pc{i+1}/umap.png')

class VolumeGenerator:
    '''Helper class to call analysis.gen_volumes'''
    def __init__(self, weights, config, vol_args={}, skip_vol=False):
        self.weights = weights
        self.config = config
        self.vol_args = vol_args
        self.skip_vol = skip_vol

    def gen_volumes(self, outdir, z_values):
        if self.skip_vol: return
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        zfile = f'{outdir}/z_values.txt'
        np.savetxt(zfile, z_values)
        analysis.gen_volumes(self.weights, self.config, zfile, outdir, **self.vol_args)

def main(args):
    t1 = dt.now()
    E = args.epoch
    log(f"analyzing {E}")
    workdir = args.workdir
    zfile = f'{workdir}/z.{E}.pkl'
    poses = args.pose
    weights = f'{workdir}/weights.{E}.pkl'
    config = f'{workdir}/config.pkl'
    outdir = f'{workdir}/analyze.{E}'
    if E == -1:
        zfile = f'{workdir}/z.pkl'
        weights = f'{workdir}/weights.pkl'
        outdir = f'{workdir}/analyze'

    if args.outdir:
        outdir = args.outdir
    log(f'Saving results to {outdir}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if args.vanilla:
        losses = analysis.parse_loss_vanilla(f"{workdir}/run.log", "validation")
        #plt.ylabel('validation loss')
        #plt.xlabel('step')
        plt.plot(np.arange(1,len(losses)+1), losses, label="validation")
        #plt.savefig(f"{workdir}/val_losses.png")
        losses = analysis.parse_loss_vanilla(f"{workdir}/run.log", "training")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(np.arange(1,len(losses)+1), losses, label="training")
        plt.xticks(range(1, len(losses)+1))
        plt.legend(loc="upper right")
        plt.savefig(f"{workdir}/train_losses.png")

        z = torch.load(zfile)["mu"].cpu().numpy()
        log("loading {}, z shape {}".format(zfile, z.shape))
        Nimg = z.shape[0]
        zdim = z.shape[1]
        posetracker = PoseTracker.load(poses, Nimg, args.D, None, None,
                                   deform=True, deform_emb_size=zdim, latents=zfile, batch_size=4)# hp_order=2)
        groups = posetracker.euler_groups
        utils.save_pkl(groups, f"{workdir}/groups.pkl")
        log("loading {}".format(poses))
    else:
        z = utils.load_pkl(zfile)
    zdim = z.shape[1]

    vol_args = dict(Apix=args.Apix, downsample=args.downsample, flip=args.flip, cuda=args.device)
    vg = VolumeGenerator(weights, config, vol_args, skip_vol=args.skip_vol)

    if zdim == 1:
        analyze_z1(z, outdir, vg)
    else:
        analyze_zN(z, outdir, vg, groups, skip_umap=args.skip_umap, num_pcs=args.pc,
                   num_ksamples=args.ksample, num_pc_samples=args.psample)

    # copy over template if file doesn't exist
    out_ipynb = f'{outdir}/cryoDRGN_viz.ipynb'
    if not os.path.exists(out_ipynb):
        log(f'Creating jupyter notebook...')
        ipynb = f'{cryodrgn._ROOT}/templates/cryoDRGN_viz_template.ipynb'
        shutil.copyfile(ipynb, out_ipynb)
    else:
        log(f'{out_ipynb} already exists. Skipping')
    log(out_ipynb)

    # copy over template if file doesn't exist
    out_ipynb = f'{outdir}/cryoDRGN_filtering.ipynb'
    if not os.path.exists(out_ipynb):
        log(f'Creating jupyter notebook...')
        ipynb = f'{cryodrgn._ROOT}/templates/cryoDRGN_filtering_template.ipynb'
        shutil.copyfile(ipynb, out_ipynb)
    else:
        log(f'{out_ipynb} already exists. Skipping')
    log(out_ipynb)

    log(f'Finished in {dt.now()-t1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
