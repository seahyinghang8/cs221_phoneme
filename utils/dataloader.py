import logging
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from normalize import get_normalizer
import numpy as np
import pickle
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class PhonemeDataset(Dataset):
    """
        Dataset of phonemes
    """
    def __init__(self, num_frames, phn_idx_map, data_type="mfcc", mode="train", transform=None):
        """
            Initialization function

            Parameters
            ----------
                - num_frames: int: number of frames
                - phn_idx_map: dict: a mapping from phoneme str to an integer, which is the label
                - data_type: str: can be mfcc, mfcc-delta, ssc, ssc-delta, logfbank, logfbank-delta
                - mode: str: 'train' or 'test'
                - transform: torch.transforms: a transform to normalize the data

        """
        self.labels = []
        self.filenames = []
        self.transform = transform
        self.data_type = data_type
        self.num_frames = num_frames
        data_dir = os.path.join(ROOT_DIR, 'data', mode)
        
        for accent in next(os.walk(data_dir))[1]:
            # first loop through the various regional accents
            acc_dir = os.path.join(data_dir, accent)
            
            for speaker in next(os.walk(acc_dir))[1]:
                # then loop through all the speakers from the region
                spk_dir = os.path.join(acc_dir, speaker)
                
                for sentence in next(os.walk(spk_dir))[1]:
                    # then loop through sentences
                    sent_dir = os.path.join(spk_dir, sentence)

                    for phoneme in next(os.walk(sent_dir))[1]:
                        if phoneme not in phn_idx_map: continue
                        # then loop through phonemes
                        phoneme_dir = os.path.join(sent_dir, phoneme)

                        for filename in os.listdir(phoneme_dir):
                            # then feature files
                            if filename.endswith("-{}.npy".format(data_type)):
                                label = phn_idx_map[phoneme]
                                self.filenames.append(os.path.join(phoneme_dir, filename))
                                self.labels.append(int(label))

        self.len = len(self.filenames)
        logging.info('Initialized a PhonemeDataset of size {0}.'.format(self.len))
        super(PhonemeDataset, self).__init__()

    def __getitem__(self, idx):
        """
            Dataloader function to get one datapoint

            Parameter
            ---------
                - idx: int: the index in the dataset
            Return
            ------
                - (feature, label): tuple: the feature and the labeled data
        """
        file_path = self.filenames[idx]
        feature = np.load(file_path)
        label = self.labels[idx]
        # convert feature into a torch tensor
        feature_t = torch.from_numpy(feature).type(torch.FloatTensor)
        # reshaping to account for the difference in channel
        old_shape = feature_t.shape
        if not self.data_type.endswith('delta'):
            feature_t = feature_t.reshape((1, old_shape[0], old_shape[1]))
        if self.transform:
            feature_t = self.transform(feature_t)
        return (feature_t, label)

    def __len__(self):
        return self.len

def get_dataloaders(batch_size, num_frames, phn_idx_map, data_type='mfcc', val_split=0.15):
    """
        Create the train, val and test dataloaders

        Parameters
        ----------
            - batch_size: int: number of examples per batch
            - num_frames: int: number of frames
            - phn_idx_map: dict: a mapping from phoneme str to an integer, which is the label
            - data_type: str: can be mfcc, mfcc-delta, logfbank_40, logfbank_40-delta
            - val_split: float: the portion of the data to be validation data
        Return
        ------
            (train_dataloader, val_dataloader, test_dataloader)

    """
    # Get the transform for each data_type
    normalize = get_normalizer(data_type)

    # Create the Datasets
    train_data = PhonemeDataset(num_frames, phn_idx_map, data_type, mode='train', transform=normalize)
    test_data = PhonemeDataset(num_frames, phn_idx_map, data_type, mode='test', transform=normalize)

    # Create DataLoaders for train, validation, test
    indices = list(range(len(train_data)))
    validation_idx = np.random.choice(indices, size=int(val_split * len(train_data)), replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    validation_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=validation_sampler)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    return train_loader, validation_loader, test_loader