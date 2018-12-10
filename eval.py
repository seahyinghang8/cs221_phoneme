import logging
import argparse
import sys
import numpy as np
import os
from tqdm import tqdm
import datetime
import cPickle as pkl

import torch

from models.cnn import cnn2Layer, cnn5Layer, cnn10Layer
from models.simple import linearRegressor

from configs.config_parser import parse

from utils.logger import setup_logging
from utils.normalize import get_normalizer
from utils.plot_cm import plot_phoneme_cm
from utils.dataloader import PhonemeDataset
from torch.utils.data import DataLoader


ARCH_TO_MODEL = {
    'linear_regressor': linearRegressor,
    'cnn_2_layer': cnn2Layer,
    'cnn_5_layer': cnn5Layer,
    'cnn_10_layer': cnn10Layer
}

def predict(model, dataloader, device, cfg):
    # generate predictions and save them
    # in addition, evaluate the loss and the error rate of the model
    model.eval()
    
    total_count = 0
    total_wrong = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            total_count += y.shape[0]
            total_wrong += sum(pred.topk(1)[1].squeeze() != y).item()

            y_pred += pred.topk(1)[1].squeeze().tolist()
            y_true += y.tolist()

    err_rate = 100. * float(total_wrong) / total_count
    return err_rate, {'y_true': y_true, 'y_pred': y_pred}

def main():
    parser = argparse.ArgumentParser(description='Phoneme classification task')
    parser.add_argument('--load', type=str, help='Path of model checkpoint to load', required=True)
    parser.add_argument('--config', '-c', type=str, help='Name of config file to load', required=True)

    args = parser.parse_args()
    setup_logging(args)
    cfg = parse(args)

    # use all gpus available
    num_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = get_normalizer(cfg.data_type)

    # get the dataloaders
    test_data = PhonemeDataset(cfg.num_frames, cfg.phn_idx_map, cfg.data_type, mode='test', transform=normalize)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=True)

    logging.info('Cmd: python {0}'.format(' '.join(sys.argv)))

    # set up model and load checkpoint
    if not cfg.model_arch:
        raise ValueError('Model architecture not specified')

    if not cfg.model_arch in ARCH_TO_MODEL:
        raise ValueError('Model architecture {} does not exist.'.format(cfg.model_arch))

    model = ARCH_TO_MODEL[cfg.model_arch](cfg.num_channels, cfg.num_frames, cfg.num_classes, cfg.num_dimensions)

    with open(os.path.join(cfg.load), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # test the model
    test_err, predictions = predict(model, test_loader, device, cfg)
    logging.info('Test PER: {:.2f}%'.format(test_err))

    # save the output into the a pickle file in the log directory
    output_dir = os.path.join(cfg.log_dir, 'predict.pkl')
    logging.info('Prediction saved in {}'.format(output_dir))

    with open(output_dir, 'w') as file:
        pkl.dump(predictions, file)

    # plot the confusion matrix
    plot_phoneme_cm(predictions, cfg.phn_idx_map)

if __name__ == '__main__':
    main()