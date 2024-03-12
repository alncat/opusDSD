# mypackage/myscript_wrapper.py
import subprocess
import os

class eval_vol:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('resdir', type=os.path.abspath, help='result directory')
        parser.add_argument('N', type=int, help='epoch number')
        parser.add_argument('method', choices=('kmeans','pc', 'dpc'), default='kmeans', help='choosing an analysis method (default: %(default)s), \
                            dpc is used to reconstruct multi-body dynamics')
        parser.add_argument('num', type=int, help='the number of KMeans clusters or PCs for reconstruction')
        parser.add_argument('apix', type=float, help='desired apix of the output volume')
        parser.add_argument('--num-bodies', default=0, type=int, required=False, help='the number of bodies defined in training (default: %(default)s)')
        parser.add_argument('--masks', type=os.path.abspath, required=False, help='path to the pkl for masks params')
        parser.add_argument('--kmeans', type=int, required=False, help='the kmeans folder to select the template to be deformed')
        parser.add_argument('--dfk', type=int, required=False, help='the kmeans center serving as the template to be deformed')
        parser.add_argument('--flip', action='store_true', required=False, help='invert handness of the reconstruction')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'eval_vol.sh')
        if args.flip:
            flip='--flip'
        else:
            flip=''
        if args.method == 'dpc':
            assert args.kmeans is not None and args.dfk is not None
            subprocess.call(['bash', script_path, args.resdir, str(args.N), args.method, str(args.num), str(args.apix), args.masks, str(args.kmeans), str(args.dfk), flip])
        else:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), args.method, str(args.num), str(args.apix), str(args.num_bodies), flip])

class analyze:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('resdir', type=os.path.abspath, help='result directory')
        parser.add_argument('N', type=int, help='epoch number to be analyzed')
        parser.add_argument('numpc', type=int, help='number of PCs')
        parser.add_argument('numk', type=int, help='number of KMeans clusters')
        parser.add_argument('--skip-umap', action='store_true', required=False, help='instead of learn a umap embedding, loading one from umap.pkl (default: %(default)s)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'analyze.sh')
        if args.skip_umap:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), str(args.numpc), str(args.numk), '--skip-umap'])
        else:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), str(args.numpc), str(args.numk),])

class parse_pose:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('resdir', type=os.path.abspath, help='result folder storing training results')
        parser.add_argument('N', type=int, help='epoch number')
        parser.add_argument('kmeans', type=int, help='KMeans clusters for classification')
        parser.add_argument('--relion31', action='store_true', help='if the input starfile is of version 3.1')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'parse_pose.sh')
        if args.relion31:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.resdir, str(args.N), str(args.kmeans), '--relion31'])
        else:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.resdir, str(args.N), str(args.kmeans),])

class prepare:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('--relion31', action='store_true', help='whether the input starfile is of version 3.1')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'prepare.sh')
        if args.relion31:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), '--relion31'])
        else:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix),])

class prepare_multi:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('masks', type=os.path.abspath, help='starfile storing mask definitions for multi-body refinement')
        parser.add_argument('numb', type=int, help='the number of bodies defined for multi-body refinement')
        parser.add_argument('--volumes', type=os.path.abspath, help='the path to the volume series generated from PCA for defining rotation axes')
        parser.add_argument('--relion31', action='store_true', help='whether the input starfile is of version 3.1')
        parser.add_argument('--outmasks', default='mask_params', help='the name of pkl file storing masks related parameters, \
                            you should omit the filetype name .pkl')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'prepare_multi.sh')
        if args.relion31:
            if args.volumes:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--volumes ' + args.volumes, '--outmasks ' + args.outmasks, '--relion31',])
            else:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '', '--outmasks ' + args.outmasks, '--relion31'])
        else:
            if args.volumes:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--volumes ' + args.volumes, '--outmasks ' + args.outmasks,])
            else:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--outmasks ' + args.outmasks,])


