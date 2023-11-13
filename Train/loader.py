# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-22
File: loader.py
"""
import os
import sys
import gc
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from data_processor import DataProcess
from logger import DotMap
sys.path.append("../Precompute")
from propagation import A2Prop


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)


def lmatstd(m):
    """Large matrix standardization"""
    rowh = m.shape[0] // 2
    std = np.std(m[:rowh], axis=0)
    m[:rowh] /= std
    m[rowh:] /= std
    gc.collect()
    return m


def matstd_clip(m, idx, with_mean=False, clip=False):
    """Standardize and clip per feature"""
    # idx = np.setdiff1d(idx, [0])
    # if (len(idx) > 0.75 * m.shape[0]) and (m.shape[0] > 2,000,000):
    #     idx = np.random.choice(idx, size=int(len(idx)/5), replace=False)
    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(m[idx])
    if clip:
        mean, std = scaler.mean_, scaler.scale_
        k = 3
        m = np.clip(m, a_min=mean-k*std, a_max=mean+k*std)
    m = scaler.transform(m)
    return m


def diag_mul(diag, m):
    """Diagonal matrix multiplication"""
    row = m.shape[0]
    for i in range(row):
        m[i] *= diag[i]
    return m


# ====================
def load_hetero_list(datastr: str, datapath: str,
                   multil: bool, chn_dct: DotMap,
                   seed: int=0, stdmean: bool=True):
    # Get degree and label
    processor = DataProcess(datastr, path=datapath, seed=seed)
    processor.input(['labels'])
    if multil:
        processor.calculate(['labels_oh'])
        labels = torch.LongTensor(processor.labels_oh)
        labels = labels.float()
    else:
        processor.labels[processor.labels < 0] = 0
        labels = torch.LongTensor(processor.labels).flatten()
    # Get index
    # processor.input(['idx_train', 'idx_val', 'idx_test'])
    processor.calculate(['idx_train'])
    idx = {'train': torch.LongTensor(processor.idx_train),
           'val':   torch.LongTensor(processor.idx_val),
           'test':  torch.LongTensor(processor.idx_test)}
    # Get graph property
    n, m = processor.n, processor.m
    # print(processor)

    def load_file(alg: str, est_name: str):
        if '-directed' in datastr and est_name.startswith('ase'):
            undatastr = '-'.join(datastr.split('-')[:-1])
            est_dir = f'../save/{undatastr}/{alg}'
        else:
            est_dir = f'../save/{datastr}/{alg}'
        est_file = f'{est_dir}/{est_name}.npy'
        # if not est_name.startswith('ase'):
        #     return None, est_file
        if os.path.exists(est_file):
            feat = np.load(est_file)
            return feat, est_file
        else:
            os.makedirs(est_dir, exist_ok=True)
            return None, est_file

    # Load feature
    idx_fit = idx['train']
    features = []
    for chn in chn_dct:
        if chn == 'attr':
            feat = np.load(os.path.join(datapath, datastr, 'feats.npy'))
            feat = matstd_clip(feat, idx_fit, with_mean=stdmean)
        elif chn.startswith('ase'):
            dct = chn_dct[chn]
            delta = dct.delta if type(dct.delta) is float else 1e-5
            dct.delta = delta
            logdelta = int(-np.log10(delta))
            est_name = f"{chn}_l{int(dct.hop):d}_m{dct.r-dct.l:g}_eps{logdelta:d}_{seed:g}"
            feat, est_file = load_file('a2prop', est_name)
            if feat is None or feat.shape[1] < dct.r - dct.l:
                print(f'Calculating {chn} {dct}...', flush=True)
                feat = np.zeros((processor.n, dct.r-dct.l), dtype=np.float32, order='C')
                py_a2prop = A2Prop()
                _ = py_a2prop.propagate(datastr, chn,
                                        processor.m, processor.n, seed,
                                        dct.hop, delta, 0, 0, 0, feat)
                np.save(est_file, feat)
            else:
                feat = feat[:, dct.l:dct.r]

            norms = np.linalg.norm(feat, axis=0)
            idxz = np.where(norms < delta/10)[0]
            if idxz.size > 0:
                rr = idxz[idxz >= max(0, feat.shape[1] - len(idxz))].min()
                feat = feat[:, :rr]
            feat = matstd_clip(feat, idx_fit, with_mean=True)
        elif chn.startswith('feat'):
            dct = chn_dct[chn]
            delta = dct.delta if type(dct.delta) is float else 1e-5
            dct.delta = delta
            logdelta = int(-np.log10(delta))
            est_name = f"{chn}_l{int(dct.hop):d}_r{dct.rrz:g}_eps{logdelta:d}_{seed:g}"
            feat, est_file = load_file('a2prop', est_name)
            if feat is None:
                print(f'Calculating {chn} {dct}...', flush=True)
                processor.input(['attr_matrix'])
                feat = processor.attr_matrix.astype(np.float32, order='C')
                py_a2prop = A2Prop()
                _ = py_a2prop.propagate(datastr, chn,
                                        processor.m, processor.n, seed,
                                        dct.hop, delta, 0, 1-dct.rrz, dct.rrz, feat)
                np.save(est_file, feat)
            feat = matstd_clip(feat, idx_fit, with_mean=True)
        else:
            raise ValueError(f'Unknown channel {chn}')
        # Append feature
        features.append(feat)

    feat = {'val':  [torch.FloatTensor(f[idx['val']]) for f in features],
            'test': [torch.FloatTensor(f[idx['test']]) for f in features]}
    feat['train'] = [torch.FloatTensor(f[idx['train']]) for f in features]
    del features
    gc.collect()
    print(f"n={n}, m={m}, label={labels.size()} | {int(labels.max())+1})")
    print([f.shape for f in feat['train']])
    return feat, labels, idx
