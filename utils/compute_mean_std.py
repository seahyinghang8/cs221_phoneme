from tqdm import tqdm
import torch
import yaml
import os
import argparse

from dataloader import PhonemeDataset
from torch.utils.data import DataLoader

NUM_FRAMES = 10

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PHN_IDX_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map.yaml')

# Calculate the mean and std for each dimension for normalization of input data
def main():
    parser = argparse.ArgumentParser(description='Compute mean and std')
    parser.add_argument('--type', '-t', type=str, help='data_type', required=True)
    args = parser.parse_args()

    num_channel = 3 if (args.type.endswith('delta')) else 1

    with open(PHN_IDX_PATH, 'r') as phn_idx_file:
        phn_idx_map = yaml.load(phn_idx_file)

    train_data = PhonemeDataset(NUM_FRAMES, phn_idx_map, args.type, mode='train')
    train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)

    totalSquare = torch.zeros(num_channel)
    total = torch.zeros(num_channel)
    count = 0

    with torch.no_grad():
        # magically acquired the dataloader
        for X, y in tqdm(train_loader):
            total += torch.sum(X, dim=(0, 2, 3))
            totalSquare += torch.sum(X ** 2, dim=(0, 2, 3))
            count += X.shape[0] * X.shape[2] * X.shape[3]
    
    mean = total / count
    std = torch.sqrt(totalSquare / count - mean ** 2)

    print("Mean for {}".format(args.type), mean)
    print("Std for {}".format(args.type), std)


if __name__ == '__main__':
    main()