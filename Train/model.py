# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-22
File: model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResLinear(nn.Module):
    def __init__(self, in_features, out_features, ftransform='none'):
        super(ResLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = None
        if ftransform == 'bn':
            self.trans_fn = nn.BatchNorm1d(out_features)
        elif ftransform == 'bias':
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.trans_fn = lambda x: x + self.bias
        elif ftransform == 'biasbn':
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bn = nn.BatchNorm1d(out_features)
            self.trans_fn = lambda x: self.bn(x + self.bias)
        else:
            self.trans_fn = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.trans_fn(output)
        # Residual connection
        if self.in_features == self.out_features:
            output += input
        return output


class Dense(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, ftransform):
        super(Dense, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(ResLinear(nfeat, nhidden, ftransform))
        for _ in range(nlayers-2):
            self.fcs.append(ResLinear(nhidden, nhidden, ftransform))
        self.fcs.append(ResLinear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x


class DenseSkip(Dense):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, ftransform):
        super(DenseSkip, self).__init__(nlayers, nfeat, nhidden, nclass, dropout, ftransform)

    def forward(self, x):
        out = F.dropout(x, self.dropout, training=self.training)
        # Shortcut connection of input features
        out1 = self.fcs[0](out)
        out = self.act_fn(out1)
        for fc in self.fcs[1:-1]:
            out = F.dropout(out, self.dropout, training=self.training)
            out = fc(out)
            out += out1
            out = self.act_fn(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fcs[-1](out)
        return out


class MLP(nn.Module):
    name = 'MLP'
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, ftransform='none'):
        super(MLP, self).__init__()
        fbias = 'bias' in ftransform
        self.fbn = 'bn' in ftransform
        self.dropout = dropout
        self.nfeat = nfeat
        self.nhidden = nhidden
        self.nclass = nclass

        self.fcs = nn.ModuleList()
        if self.fbn: self.bns = nn.ModuleList()

        if nlayers == 1:
            self.fcs.append(nn.Linear(nfeat, nclass, bias=fbias))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden, bias=fbias))
            if self.fbn: self.bns.append(nn.BatchNorm1d(nhidden))
            for _ in range(nlayers - 2):
                self.fcs.append(nn.Linear(nhidden, nhidden, bias=fbias))
                if self.fbn: self.bns.append(nn.BatchNorm1d(nhidden))
            self.fcs.append(nn.Linear(nhidden, nclass, bias=fbias))

    def reset_parameters(self):
        for lin in self.fcs:
            lin.reset_parameters()
        if self.fbn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            x = F.relu(x, inplace=True)
            if self.fbn: x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x


class FusionChn(nn.Module):
    """Seperate MLP for each channel"""
    name = 'FusionChn'
    def __init__(self, nflayers, nclayers, nchn, nfeat, nhidden, nclass, dropout, ftransform, fusion='cat'):
        super(FusionChn, self).__init__()
        # Feature embed
        self.nchn = nchn
        if type(nfeat) == int:
            nfeat = [nfeat] * nchn
        self.feat = nn.ModuleList()
        for chn in range(nchn):
            self.feat.append(MLP(nflayers, nfeat[chn], nhidden, nhidden,
                                 dropout=0, ftransform=ftransform.replace('bn','')))
        # Fusion layer
        self.fusion = None
        if fusion == 'cat':
            nfusion = nchn*nhidden
            self.fusion_fn = lambda x: F.relu(torch.cat(x, axis=-1), inplace=True)
        elif fusion == 'add':
            nfusion = nhidden
            self.fusion_fn = lambda x: F.relu(torch.sum(torch.stack(x), axis=0), inplace=True)
        elif fusion == 'ladd':
            nfusion = nhidden
            self.fusion = nn.Parameter(torch.ones(nchn))
            self.fusion_fn = lambda x: self._fusion_ladd(x)
        elif fusion == 'catadd':
            nfusion = nhidden
            self.fusion = ResLinear(nchn*nhidden, nhidden, ftransform='none')
            self.fusion_fn = lambda x: self._fusion_catadd(x)
        else:
            raise NotImplementedError('Fusion type not implemented: {}'.format(fusion))

        # Classifier layers
        self.classifier = MLP(nclayers, nfusion, nhidden, nclass,
                              dropout=dropout, ftransform=ftransform)
        # self.classifier = Dense(nclayers, nfusion, nhidden, nclass,
        #                         dropout=dropout, ftransform=ftransform)

    def _fusion_catadd(self, out: list):
        """LINKX-like fusion: sigma(W[Pl, Ph] + Pl + Ph)"""
        x = torch.cat(out, axis=-1)
        x = self.fusion(x)
        out.append(x)
        x = torch.sum(torch.stack(out), axis=0)
        x = F.relu(x, inplace=True)
        return x

    def _fusion_ladd(self, out: list):
        """Lanczos fusion: sigma(thetal * Pl + thetah * Ph))"""
        x = torch.zeros_like(out[0])
        for i in range(self.nchn):
            x += out[i] * self.fusion[i]
        x = F.relu(x, inplace=True)
        return x

    def reset_parameters(self):
        for feat in self.feat:
            feat.reset_parameters()
        if self.fusion is not None:
            self.fusion.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x):
        out = []
        # for i in range(self.nchn):
        #     out.append(self.feat[i](x[:,i,:self.feat[i].nfeat]))
        for i, xi in enumerate(x):
            out.append(self.feat[i](xi))
        x = self.fusion_fn(out)
        x = self.classifier(x)
        return x


class FusionAX(nn.Module):
    name = 'FusionAX'
    def __init__(self, nflayers, nclayers, nchn, nfeat, nhidden, nclass, dropout, ftransform, fusion='cat'):
        super(FusionAX, self).__init__()
        # Feature embed
        self.nchn = nchn
        if type(nfeat) == int:
            nfeat = [nfeat] * nchn
        self.feat = nn.ModuleList()
        for chn in range(nchn):
            self.feat.append(MLP(nflayers, nfeat[chn], nhidden, nhidden,
                                 dropout=0, ftransform=ftransform.replace('bn','')))
        # Fusion layer
        nfusion = nhidden
        if fusion == 'cat':
            self.axadd = False
        else:
            self.axadd = True
        self.fusionX = ResLinear((nchn-1)*nhidden, nhidden, ftransform='none')
        self.fusion = ResLinear(2*nhidden, nhidden, ftransform='none')
        self.fusion_fn = lambda x: self._fusion_catadd(x)

        # Classifier layers
        self.classifier = MLP(nclayers, nfusion, nhidden, nclass,
                              dropout=dropout, ftransform=ftransform)
        # self.classifier = Dense(nclayers, nfusion, nhidden, nclass,
        #                         dropout=dropout, ftransform=ftransform)

    def _fusion_catadd(self, out: list):
        """LINKX-like fusion: MLP(sigma(W[Pl, Ph] + Pl + Ph))"""
        emb_a = out[0]
        x = torch.cat(out[1:], axis=-1)
        x = self.fusionX(x)
        # out.append(x)
        # x = torch.sum(torch.stack(out[1:]), axis=0)
        x = F.relu(x, inplace=True)
        out = [emb_a, x]
        x = torch.cat(out, axis=-1)
        x = self.fusion(x)
        if self.axadd:
            out.append(x)
            x = torch.sum(torch.stack(out), axis=0)
        x = F.relu(x, inplace=True)
        return x

    def reset_parameters(self):
        for feat in self.feat:
            feat.reset_parameters()
        if self.fusion is not None:
            self.fusion.reset_parameters()
            self.fusionX.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x):
        out = []
        for i, xi in enumerate(x):
            out.append(self.feat[i](xi))
        x = self.fusion_fn(out)
        x = self.classifier(x)
        return x
