from tqdm import tqdm
import yaml
import os
import argparse
import torch
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.decomposition import PCA

from dataloader import PhonemeDataset
from torch.utils.data import DataLoader

# Plot phonemes in reduced feature space

NUM_FRAMES = 10

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PHN_IDX_PATH     = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map.yaml')
IDX_PHN_PATH     = os.path.join(ROOT_DIR, 'phn_index_maps', 'ls_map.yaml')
PHN_IDX_ORD_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map_ordered.yaml')
IDX_PHN_ORD_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'ls_map_ordered.yaml')
PHN_IDX_GRP_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'phn_index_map_ordered_grouped.yaml')
IDX_PHN_GRP_PATH = os.path.join(ROOT_DIR, 'phn_index_maps', 'ls_map_ordered_grouped.yaml')

def main():
    parser = argparse.ArgumentParser(description='Combine phonemes into a pickle file')
    parser.add_argument('--type', '-t', type=str, help='data_type', required=True)
    args = parser.parse_args()

    with open(PHN_IDX_PATH, 'r') as phn_idx_file:
        phn_idx_map = yaml.load(phn_idx_file)

    with open(IDX_PHN_PATH, 'r') as idx_phn_file:
        idx_phn_map = yaml.load(idx_phn_file)

    with open(PHN_IDX_ORD_PATH, 'r') as phn_idx_file:
        phn_idx_ord_map = yaml.load(phn_idx_file)

    with open(IDX_PHN_ORD_PATH, 'r') as idx_phn_file:
        idx_phn_ord_map = yaml.load(idx_phn_file)

    with open(PHN_IDX_GRP_PATH, 'r') as phn_idx_file:
        phn_idx_grp_map = yaml.load(phn_idx_file)

    with open(IDX_PHN_GRP_PATH, 'r') as idx_phn_file:
        idx_phn_grp_map = yaml.load(idx_phn_file)

    try:
        print "Checking if pickled dataset exists"
        data = pickle.load(open("{}_1_frame.pkl".format(args.type), "rb"))
        labels = pickle.load(open("labels.pkl", "rb"))
    except:
        print "Loading dataset"
        train_data = PhonemeDataset(NUM_FRAMES, phn_idx_map, args.type, mode='train')
        data = np.array([train_data[i][0][0][NUM_FRAMES / 2 - 1].numpy().flatten() for i in range(len(train_data))])
        labels = np.array([train_data[i][1] for i in range(len(train_data))])
        pickle.dump(data, open("{}_1_frame.pkl".format(args.type, NUM_FRAMES), "wb"))
        pickle.dump(labels, open("labels.pkl", "wb"))

    phoneme_colors = np.array([phn_idx_ord_map[idx_phn_map[i][0]] for i in labels])
    group_colors = np.array([phn_idx_grp_map[idx_phn_map[i][0]] for i in labels])

    print "Performing PCA"
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)

    feature_type_dir = os.path.join(ROOT_DIR, '{}-features-2d-plot'.format(args.type))
    if not os.path.exists(feature_type_dir):
        os.mkdir(feature_type_dir)

    print "Plotting features"
    plt.scatter(proj[:, 0], proj[:, 1], s=1, c=group_colors, cmap='gist_rainbow')
    c = plt.colorbar()
    plt.savefig(os.path.join(feature_type_dir, "{}-pca-phn".format(args.type)), dpi=100)

    c.remove()
    plt.scatter(proj[:, 0], proj[:, 1], s=1, c=phoneme_colors, cmap='gist_rainbow')
    c = plt.colorbar()
    plt.savefig(os.path.join(feature_type_dir, "{}-pca-grp".format(args.type)), dpi=100)

if __name__ == '__main__':
    main()