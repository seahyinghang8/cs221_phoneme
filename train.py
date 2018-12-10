import logging
import argparse
import sys
import numpy as np
import os
from tqdm import tqdm
import datetime

import torch
from torch.optim import Adam, SGD
from torch.nn.modules.loss import NLLLoss

from models.cnn import cnn2Layer, cnn5Layer, cnn10Layer
from models.simple import linearRegressor

from configs.config_parser import parse

from utils.sampler import sample_cfg
from utils.logger import setup_logging
from utils.dataloader import get_dataloaders


ARCH_TO_MODEL = {
    'linear_regressor': linearRegressor,
    'cnn_2_layer': cnn2Layer,
    'cnn_5_layer': cnn5Layer,
    'cnn_10_layer': cnn10Layer
}


def save_model(model, path, best=False):
    model_path = os.path.join(path, 'best_model.pth' if best else 'latest_model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().module.state_dict(), model_path)

def evaluate(model, dataloader, loss_fn, device, cfg):
    # evaluate the loss and the error rate of the model
    model.eval()
    total_loss = 0.0
    
    total_count = 0
    total_wrong = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            total_count += y.shape[0]
            total_wrong += sum(pred.topk(1)[1].squeeze() != y).item()
            loss = loss_fn(pred, y)
            total_loss += loss.item()

    err_rate = 100. * float(total_wrong) / total_count
    avg_loss = float(total_loss) / total_count
    return avg_loss, err_rate

def train(model, dataloader, loss_fn, device, cfg, optimizer):
    # trains the model given a loss function and an optimizer
    model.train()
    total_loss = 0.

    total_count = 0
    total_wrong = 0

    for X, y in tqdm(dataloader):
        optimizer.zero_grad()
        
        X, y = X.to(device), y.to(device)
        pred = model(X)

        total_count += y.shape[0]
        total_wrong += sum(pred.topk(1)[1].squeeze() != y).item()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    err_rate = 100. * float(total_wrong) / total_count
    avg_loss = float(total_loss) / total_count
    return avg_loss, err_rate

def main():
    parser = argparse.ArgumentParser(description='Phoneme classification task')
    parser.add_argument('--config', '-c', type=str, help='Name of config file to load', required=True)

    args = parser.parse_args()
    setup_logging(args)
    cfg = parse(args)

    # use all gpus available
    num_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'gpu': logging.info('Look ma, I\'m using a GPU!')

    # get the dataloaders
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(cfg.batch_size, cfg.num_frames, cfg.phn_idx_map, cfg.data_type)

    # setup hyperparameter sweeping
    if not cfg.num_sweeps:
        cfg.num_sweeps = 1

    logging.info('Cmd: python {0}'.format(' '.join(sys.argv)))
    logging.info('Running experiment <{0}> in train mode.'.format(cfg.config, cfg.mode))

    orig_cfg = cfg
    for sweep_count in range(cfg.num_sweeps):
        cfg = sample_cfg(orig_cfg)
        # create a directory for the model to be saved
        cfg.sweep_dir = os.path.join(cfg.log_dir, 's_{}'.format(sweep_count))
        os.mkdir(cfg.sweep_dir)
        logging.info('Sweep Count: {}'.format(sweep_count))
        logging.info('Config:\n {0}'.format(cfg))
        # set up model and load if checkpoint provided
        if not cfg.model_arch:
            raise ValueError('Model architecture not specified')

        if not cfg.model_arch in ARCH_TO_MODEL:
            raise ValueError('Model architecture {} does not exist.'.format(cfg.model_arch))

        model = ARCH_TO_MODEL[cfg.model_arch](cfg.num_channels, cfg.num_frames, cfg.num_classes, cfg.num_dimensions)

        if cfg.load:
            with open(os.path.join(cfg.load), 'rb') as f:
                model.load_state_dict(torch.load(f))
        model = model.to(device)
        model = torch.nn.DataParallel(model)

        # setup cross entropy loss for multi-class classification
        loss = NLLLoss()

        # setup the optimizer
        optim = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularizer)

        # train the model
        logging.info('Commencing training')
        lowest_err = 100.
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_err = train(model, train_dataloader, loss, device, cfg, optim)
            val_loss, val_err = evaluate(model, valid_dataloader, loss, device, cfg)
            logging.info('Epoch: {}\tTrain Loss: {:.6f}\tTrain PER: {:.2f}\tVal Loss: {:.6f}\tVal PER: {:.2f}%'
                .format(epoch, train_loss, train_err, val_loss, val_err))
            if val_err < lowest_err:
                logging.info('Lowest validation error achieved. Saving the model.')
                save_model(model, cfg.sweep_dir, best=True)
                lowest_err = val_err
        # save the checkpoint of the latest version of the model
        save_model(model, cfg.sweep_dir)
        # test the model
        _, test_err = evaluate(model, test_dataloader, loss, device, cfg)
        logging.info('Test PER: {:.2f}%'.format(test_err))

if __name__ == '__main__':
    main()
