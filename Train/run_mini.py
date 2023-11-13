# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-05-03
File: run_mini.py
"""
import time
import random
import argparse
import resource
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from logger import Logger, ModelLogger, prepare_opt, ScoreCalculator, get_num_params, get_mem_params
from loader import load_hetero_list
import model

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=7, help='random seed.')
parser.add_argument('-c', '--config', default='./config/arxiv.json', help='config path.')
parser.add_argument('-v', '--dev', type=int, default=0, help='device id.')
parser.add_argument('-n', '--suffix', default='', help='name suffix')
args = prepare_opt(parser)

num_thread = 32
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    torch.cuda.manual_seed(args.seed)

# print('-' * 20)
flag_run = str(args.seed)
logger = Logger(args.data, args.algo, flag_run=flag_run)
print(args.toDict())
# logger.save_opt(args)
model_logger = ModelLogger(logger, prefix='model'+args.suffix, state_only=True)

feat, labels, idx = load_hetero_list(datastr=args.data, datapath=args.path,
            multil=args.multil, chn_dct=args.chn, seed=args.seed)
nfeat = [feat['train'][i].shape[1] for i, _ in enumerate(args.chn)]
nclass = labels.shape[1] if args.multil else int(labels.max()) + 1

model = model.FusionChn(nflayers=1, nclayers=args.layer-1, nchn=len(args.chn),
                        nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                        dropout=args.dropout, ftransform=args.bias, fusion=args.fusion)
print(model)
model_logger.regi_model(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=6, verbose=False)
loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()

ds_train = Data.TensorDataset(*feat['train'], labels[idx['train']])
loader_train = Data.DataLoader(dataset=ds_train, batch_size=args.batch,
                               shuffle=True, num_workers=num_thread)
ds_val = Data.TensorDataset(*feat['val'], labels[idx['val']])
loader_val = Data.DataLoader(dataset=ds_val, batch_size=args.batch,
                             shuffle=False, num_workers=num_thread)
ds_test = Data.TensorDataset(*feat['test'], labels[idx['test']])
loader_test = Data.DataLoader(dataset=ds_test, batch_size=args.batch,
                              shuffle=False, num_workers=num_thread)


def train(ld=loader_train):
    model.train()
    loss_list = []
    time_epoch = 0
    for _, batch in enumerate(ld):
        batch_xs, batch_y = batch[:-1], batch[-1]
        if args.dev >= 0:
            batch_xs = [batch_x.cuda(args.dev) for batch_x in batch_xs]
            batch_y = batch_y.cuda(args.dev)

        time_start = time.time()
        optimizer.zero_grad()
        output = model(batch_xs)
        loss_batch = loss_fn(output, batch_y)
        loss_batch.backward()
        optimizer.step()
        time_epoch += (time.time()-time_start)

        loss_list.append(loss_batch.item())
    return np.mean(loss_list), time_epoch


def eval(ld):
    model.eval()
    time_epoch = 0
    calc = ScoreCalculator(nclass)
    output_l, labels_l = None, None
    with torch.no_grad():
        for step, batch in enumerate(ld):
            batch_xs, batch_y = batch[:-1], batch[-1]
            if args.dev >= 0:
                batch_xs = [batch_x.cuda(args.dev) for batch_x in batch_xs]
                batch_y = batch_y.cuda(args.dev)

            time_start = time.time()
            output = model(batch_xs)
            time_epoch += (time.time()-time_start)

            output = output.detach()
            batch_y = batch_y.detach()
            if args.multil:
                output = torch.where(output > 0, torch.tensor(1, device=output.device), torch.tensor(0, device=output.device))
            else:
                output = output.argmax(dim=1)
            calc.update(batch_y, output)

            output = output.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()
            output_l = output if output_l is None else np.concatenate((output_l, output), axis=0)
            labels_l = batch_y if labels_l is None else np.concatenate((labels_l, batch_y), axis=0)
    if args.multil:
        res = calc.compute('micro')
        # res = calc.compute('macro')
    else:
        res = calc.compute('micro')
    return res, time_epoch, output_l, labels_l


# print('-' * 20, flush=True)
# print('Start training...')
train_time = 0
conv_epoch, acc_best = 0, 0

for epoch in range(args.epochs):
    loss_train, train_ep = train()
    train_time += train_ep
    acc_val, _, _, _ = eval(ld=loader_val)
    scheduler.step(acc_val)
    if (epoch+1) % 1 == 0:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, cost:{train_time:.4f}"
        print(res)
        # logger.print(res)
    is_best = (acc_val > acc_best)
    if is_best:
        model_logger.save_mem()
        acc_best = acc_val
    # Early stop if converge
    conv_epoch = 0 if is_best else conv_epoch + 1
    if conv_epoch == args.patience:
        conv_epoch = epoch
        break

model = model_logger.load_mem()
if args.dev >= 0:
    model = model.cuda(args.dev)

# print('-' * 20)
# print("Start inference...")
start = time.time()
acc_test, time_inf, outl, labl = eval(ld=loader_test)
time_eval = time.time() - start
print(f'Test acc: {acc_test:.4f}', flush=True)

memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mem_cuda = torch.cuda.max_memory_reserved(args.dev)
acc_train, _, _, _ = eval(ld=loader_train)
print(f"Train time cost: {train_time:0.4f}, Best epoch: {conv_epoch}, Epoch avg: {train_time*1000 / (epoch+1):0.1f}")
print(f"Train best acc: {acc_train:0.4f}, Val best acc: {acc_best:0.4f}", flush=True)
print(f"Test time cost: {time_inf:0.4f}, eval time: {time_eval:0.4f}, RAM: {memory / 2**20:.3f} GB, CUDA: {mem_cuda / 2**30:.3f} GB")
print(f"Num params (M): {get_num_params(model):0.4f}, Mem params (MB): {get_mem_params(model):0.4f}")
