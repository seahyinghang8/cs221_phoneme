from tqdm import tqdm
import yaml
import os
import argparse
import cPickle as pkl
import torch

from collections import defaultdict
from dataloader import PhonemeDataset
from torch.utils.data import DataLoader

# Count the each number of phoneme for a pickle file

NUM_FRAMES = 10

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = os.path.dirname(PROJ_DIR)
PHN_IDX_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map.yaml')
IDX_PHN_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'ls_map.yaml')

def main():
    parser = argparse.ArgumentParser(description='Combine phonemes into a pickle file')
    parser.add_argument('--type', '-t', type=str, help='data_type', required=True)
    args = parser.parse_args()

    with open(PHN_IDX_PATH, 'r') as phn_idx_file:
        phn_idx_map = yaml.load(phn_idx_file)

    with open(IDX_PHN_PATH, 'r') as idx_phn_file:
        idx_phn_map = yaml.load(idx_phn_file)

    train_data = PhonemeDataset(NUM_FRAMES, phn_idx_map, args.type, mode='train')
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    phoneme_count = defaultdict(int)

    with torch.no_grad():
        # magically acquired the dataloader
        for X, y in tqdm(train_loader):
            x = X.squeeze().numpy()
            phn = idx_phn_map[y.item()][0]

            phoneme_count[phn] += 1

    print(phoneme_count)

    with open('count.pkl', 'wb') as output_file:
        pkl.dump(phoneme_count, output_file)


if __name__ == '__main__':
    main()