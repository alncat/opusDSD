'''
Lightweight parser for starfiles
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt
import os

from . import mrc
from .mrc import LazyImage

class Starfile():

    def __init__(self, headers, df, multibodies=None, multibody_headers=None, optics=None, relion31=False):
        assert headers == list(df.columns), f'{headers} != {df.columns}'
        self.headers = headers
        self.df = df
        self.multibodies = multibodies
        self.multibody_headers = multibody_headers
        if optics is not None:
            self.optics_header = optics[0]
            self.optics_df = optics[1]
        self.relion31 = relion31

    def __len__(self):
        return len(self.df)

    @classmethod
    def get_block(self, f, block_name):
        # get to data block
        block_found = False
        while 1:
            for line in f:
                if line.startswith(block_name):
                    block_found = True
                    break
            if not block_found:
                return "", None
            break
        # get to header loop
        while 1:
            for line in f:
                if line.startswith('loop_'):
                    break
            break
        # get list of column headers
        while 1:
            headers = []
            for line in f:
                if line.startswith('_'):
                    headers.append(line)
                else:
                    break
            break
        # assume all subsequent lines until empty line is the body
        headers = [h.strip().split()[0] for h in headers]
        body = [line]
        for line in f:
            if line.strip() == '':
                break
            body.append(line)
        # put data into an array and instantiate as dataframe
        words = [l.strip().split() for l in body]
        words = np.array(words)
        assert words.ndim == 2, f"Uneven # columns detected in parsing {set([len(x) for x in words])}. Is this a RELION 3.1 starfile?"
        assert words.shape[1] == len(headers), f"Error in parsing. Number of columns {words.shape[1]} != number of headers {len(headers)}"
        data = {h:words[:,i] for i,h in enumerate(headers)}
        df = pd.DataFrame(data=data)
        return headers, df


    @classmethod
    def load_multibody(self, starfile, relion31=False):
        f = open(starfile,'r')
        # get to data block
        if relion31:
            optics_header, optics_df = Starfile.get_block(f, 'data_optics')
            print(optics_header, optics_df)
        BLOCK = 'data_particles' if relion31 else 'data_'
        headers, df = Starfile.get_block(f, BLOCK)
        multibodies = []
        multibody_headers = []
        while 1:
            header, df_tmp = Starfile.get_block(f, 'data_images_body')
            if header == "":
                break
            print(header)
            multibodies.append(df_tmp)
            multibody_headers.append(header)
        if relion31:
            return self(headers, df, multibodies=multibodies, multibody_headers=multibody_headers, optics=(optics_header, optics_df), relion31=relion31)
        else:
            return self(headers, df, multibodies=multibodies, multibody_headers=multibody_headers, relion31=relion31)

    @classmethod
    def load(self, starfile, relion31=False):
        f = open(starfile,'r')
        if relion31:
            optics_header, optics_df = Starfile.get_block(f, 'data_optics')
            print(optics_header, optics_df)
        # get to data block
        BLOCK = 'data_particles' if relion31 else 'data_'
        while 1:
            for line in f:
                if line.startswith(BLOCK):
                    break
            break
        # get to header loop
        while 1:
            for line in f:
                if line.startswith('loop_'):
                    break
            break
        # get list of column headers
        while 1:
            headers = []
            for line in f:
                if line.startswith('_'):
                    headers.append(line)
                else:
                    break
            break
        # assume all subsequent lines until empty line is the body
        headers = [h.strip().split()[0] for h in headers]
        body = [line]
        for line in f:
            if line.strip() == '':
                break
            body.append(line)
        # put data into an array and instantiate as dataframe
        words = [l.strip().split() for l in body]
        words = np.array(words)
        assert words.ndim == 2, f"Uneven # columns detected in parsing {set([len(x) for x in words])}. Is this a RELION 3.1 starfile?"
        assert words.shape[1] == len(headers), f"Error in parsing. Number of columns {words.shape[1]} != number of headers {len(headers)}"
        data = {h:words[:,i] for i,h in enumerate(headers)}
        df = pd.DataFrame(data=data)
        if relion31:
            return self(headers, df, optics=(optics_header, optics_df), relion31=relion31)
        else:
            return self(headers, df, optics=None, relion31=relion31)

    def write_block(self, f, header, df, block_name):
        f.write(block_name + '\n\n')
        f.write('loop_\n')
        f.write('\n'.join(header))
        f.write('\n')
        for i in df.index:
            f.write(' '.join([str(v) for v in df.loc[i]]))
            f.write('\n')
        f.write('\n')

    def write(self, outstar):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        if self.relion31:
            self.write_block(f, self.optics_header, self.optics_df, 'data_optics')
        if self.relion31:
            f.write('data_particles\n\n')
        else:
            f.write('data_\n\n')
        f.write('loop_\n')
        f.write('\n'.join(self.headers))
        f.write('\n')
        for i in self.df.index:
            # TODO: Assumes header and df ordering is consistent
            f.write(' '.join([str(v) for v in self.df.loc[i]]))
            f.write('\n')
        #f.write('\n'.join([' '.join(self.df.loc[i]) for i in range(len(self.df))]))

    def write_subset(self, outstar, label):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        if self.relion31:
            self.write_block(f, self.optics_header, self.optics_df, 'data_optics')
        if self.relion31:
            f.write('data_particles\n\n')
        else:
            f.write('data_\n\n')
        f.write('loop_\n')
        f.write('\n'.join(self.headers))
        f.write('\n')
        for i in self.df.index:
            if label[i]:
                # TODO: Assumes header and df ordering is consistent
                f.write(' '.join([str(v) for v in self.df.loc[i]]))
                f.write('\n')

    def get_particles(self, datadir=None, lazy=True):
        '''
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        particles = self.df['_rlnImageName']

        # format is index@path_to_mrc
        particles = [x.split('@') for x in particles]
        ind = [int(x[0])-1 for x in particles] # convert to 0-based indexing
        mrcs = [x[1] for x in particles]
        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'
        header = mrc.parse_header(mrcs[0])
        D = header.D # image size along one dimension in pixels
        dtype = header.dtype
        stride = dtype().itemsize*D*D
        dataset = [LazyImage(f, (D,D), dtype, 1024+ii*stride) for ii,f in zip(ind, mrcs)]
        if not lazy:
            dataset = np.array([x.get() for x in dataset])
        return dataset

def prefix_paths(mrcs, datadir):
    mrcs1 = ['{}/{}'.format(datadir, os.path.basename(x)) for x in mrcs]
    mrcs2 = ['{}/{}'.format(datadir, x) for x in mrcs]
    try:
        for path in set(mrcs1):
            assert os.path.exists(path)
        mrcs = mrcs1
    except:
        for path in set(mrcs2):
            assert os.path.exists(path), f'{path} not found'
        mrcs = mrcs2
    return mrcs

def csparc_get_particles(csfile, datadir=None, lazy=True):
    metadata = np.load(csfile)
    ind = metadata['blob/idx'] # 0-based indexing
    mrcs = metadata['blob/path'].astype(str).tolist()
    if datadir is not None:
        mrcs = prefix_paths(mrcs, datadir)
    for path in set(mrcs):
        assert os.path.exists(path), f'{path} not found'
    D = metadata[0]['blob/shape'][0]
    dtype = np.float32
    stride = np.float32().itemsize*D*D
    dataset = [LazyImage(f, (D,D), dtype, 1024+ii*stride) for ii,f in zip(ind, mrcs)]
    if not lazy:
        dataset = np.array([x.get() for x in dataset])
    return dataset




