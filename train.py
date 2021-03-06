import argparse

import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from dataset.dataset import BioData

import models.BioModelCnn
from logger import TermLogger, AverageMeter

parser = argparse.ArgumentParser(description='Bio data experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('root', metavar='DIR', help='path to dataset')
parser.add_argument('--seed', default=100, type=int, help='seed set to random')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--evaluate', action='store_true', help='If it is in evaluate mode')
parser.add_argument('--print_freq', default=100, help='The frequency of printing results')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epochs = 0

    train_set = BioData(args.root, seed=args.seed, train=True)
    val_set = BioData(args.root, seed=args.seed, train=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    bio_net = models.BioModelCnn.BioModelCnn().to(device)
    bio_net.init_weights()

    cudnn.benchmark = True
    bio_net = torch.nn.DataParallel(bio_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': bio_net.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, bio_net, optimizer, args.epoch_size, logger)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
        logger.reset_valid_bar()

    logger.epoch_bar.finish()


def train(args, train_loader, bio_net, optimizer, epoch_size, logger):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    bio_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (sample, value) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data = sample.to(device)

        # compute output
        estimated_y = bio_net(data)
        estimated_y = estimated_y.view(-1)

        value = value.float()
        value = value.to(device)
        loss = value - estimated_y
        # print("value is:", value.size())
        # print("estimated_y:", estimated_y.size())

        # record loss and EPE
        # print("loss", loss.size())
        # print("loss.item:", loss)

        loss_sum = torch.sum(loss.data)
        loss_sum = Variable(loss_sum, requires_grad=True)
        # print("loss.item:", loss_sum, loss_sum.item())
        # print("args.batch_size", args.batch_size)
        losses.update(loss_sum.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


if __name__ == '__main__':
    main()
