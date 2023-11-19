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


def dmap2dct(chnname: str, dmap: DotMap):
    typedct = {'aseadj': -1, 'aseadj2': -2,
               'featadj': 0, 'featadji': 1, 'featadj2': 2, 'featadj2i': 3,
               'featlap': 4, 'featlapi': 5, 'featlap2': 6, 'featlap2i': 7,}

    dct = {}
    dct['type'] = typedct[chnname]
    dct['L'] = dmap.hop
    dct['rmax'] =  dmap.delta if type(dmap.delta) is float else 1e-5
    dct['alpha'] = dmap.alpha if type(dmap.alpha) is float else 0
    dct['rra'] = (1 - dmap.rrz) if type(dmap.rrz) is float else 0
    dct['rrb'] = dmap.rrz if type(dmap.rrz) is float else 0
    return dct


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

    # Load feature
    idx_fit = idx['train']
    features = []
    time_pre = 0
    for chnk, chnv in chn_dct.items():
        if chnk == 'attr':
            feat = np.load(os.path.join(datapath, datastr, 'feats.npy'))
            feat = matstd_clip(feat, idx_fit, with_mean=stdmean)
        # NOTE: rewrite to unify feature and call only once
        elif chnk.startswith('ase'):
            chns = [dmap2dct(chnk, DotMap(chnv))]
            nchn = len(chns)

            feat = np.zeros((chnv.r-chnv.l, processor.n), dtype=np.float32, order='C')
            feat = np.repeat(feat, nchn, axis=0)
            feat = np.ascontiguousarray(feat)
            py_a2prop = A2Prop()
            time_pre += py_a2prop.propagatea(os.path.join(datapath, datastr), chns,
                                             processor.m, processor.n, nchn, seed, feat)

            feat = feat.transpose()
            norms = np.linalg.norm(feat, axis=0)
            idxz = np.where(norms < chns[0]['rmax']/10)[0]
            if idxz.size > 0:
                rr = idxz[idxz >= max(0, feat.shape[1] - len(idxz))].min()
                feat = feat[:, :rr]
            feat = matstd_clip(feat, idx_fit, with_mean=stdmean)
        elif chnk.startswith('feat'):
            chns = [dmap2dct(chnk, DotMap(chnv))]
            nchn = len(chns)

            processor.input(['attr_matrix', 'deg'])
            feat = processor.attr_matrix.transpose().astype(np.float32, order='C')
            nfeat = feat.shape[0]
            feat = np.repeat(feat, nchn, axis=0)

            for c, chn in enumerate(chns):
                deg_b = np.power(np.maximum(processor.deg, 1e-12), chn['rrb'])
                idx_zero = np.where(deg_b == 0)[0]
                assert idx_zero.size == 0, f"Isolated nodes found: {idx_zero}"
                # deg_b[idx_zero] = 1
                feat[c*nfeat:(c+1)*nfeat, :] /= deg_b

            feat = np.ascontiguousarray(feat)
            py_a2prop = A2Prop()
            time_pre += py_a2prop.propagatea(os.path.join(datapath, datastr), chns,
                                             processor.m, processor.n, nchn, seed, feat)

            for c, chn in enumerate(chns):
                feat[c*nfeat:(c+1)*nfeat, :] *= deg_b
            feat = feat.transpose()
            feat = matstd_clip(feat, idx_fit, with_mean=stdmean)
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
    return feat, labels, idx, time_pre
