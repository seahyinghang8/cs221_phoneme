from tqdm import tqdm
import yaml
import os
import argparse
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dataloader import PhonemeDataset
from torch.utils.data import DataLoader

# For all 60 phonemes, save 12 examples into an image in the root folder

NUM_FRAMES = 10

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PHN_IDX_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map.yaml')
IDX_PHN_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'ls_map.yaml')

type_to_min_max = {
    'mfcc': (-20, 20),
    'logfbank': (0, 20),
    'logfbank_40': (0, 20)
}


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

    all_data = {}

    with torch.no_grad():
        # magically acquired the dataloader
        for X, y in tqdm(train_loader):
            x = X.squeeze().numpy()
            phn = idx_phn_map[y.item()][0]

            if phn not in all_data: all_data[phn] = []
            if len(all_data[phn]) >= 12: continue

            all_data[phn].append(x)

    vmin, vmax = type_to_min_max[args.type]

    # save the images to the root dir
    feature_type_dir = os.path.join(ROOT_DIR, '{}-features-plot'.format(args.type))
    if not os.path.exists(feature_type_dir):
        os.mkdir(feature_type_dir)

    for phn in all_data:
        phn_list = all_data[phn]
        plt.suptitle(phn)

        for i, phn_feature in enumerate(phn_list):
            plt.subplot(3, 4, i + 1)
            phn_feature = phn_feature.swapaxes(0, 1)
            plt.imshow(phn_feature, cmap='Greys', aspect='auto', vmin=vmin, vmax=vmax)

        plt.savefig(os.path.join(feature_type_dir, '{}.png'.format(phn)))

if __name__ == '__main__':
    main()