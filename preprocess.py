# This is a script that extracts the feature from TIMIT audio files
import os
import glob
import random
import numpy as np
import scipy.io.wavfile as wav
import wave
from python_speech_features import mfcc, logfbank, ssc

# Fixed Variables for FFT
WIN_LENGTH = 0.02
WIN_STEP = 0.005
MIN_DATAPOINTS = 1500
NUM_FRAMES = 10

# Setup directories
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TIMIT_DIR = os.path.join(ROOT_DIR, 'timit')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def extract_features(phn_wav_filename, base_filename):
    """
        Extract the features such as MFCC, Log Filter Bank and save the 
        features into the data directory

        Parameter
        ---------
            phn_wav_filename: str: file name with the wav file
            base_filename: str: base filename for the feature files to be stored under
        Return
        ------
            None
    """
    (rate, sig) = wav.read(phn_wav_filename)

    for extracted_feature, fn_name in zip([mfcc(sig, rate, winlen=WIN_LENGTH, winstep=WIN_STEP),
                                          logfbank(sig, rate, winlen=WIN_LENGTH, winstep=WIN_STEP, nfilt=40)],
                                          ['mfcc', 'logfbank_40']):

        if extracted_feature.shape[0] < NUM_FRAMES:
            print('Skipping phoneme as the MFCC has less frames than required')
            continue

        # If audio files need 
        start_index = extracted_feature.shape[0]/2 - NUM_FRAMES / 2
        sample_feature = extracted_feature[start_index: start_index + NUM_FRAMES]

        filename = '{}-{}.npy'.format(base_filename, fn_name)
        np.save(filename, sample_feature)

        #calc first order derivatives
        gradf = np.gradient(sample_feature)[0]
        #calc second order derivatives
        grad2f = np.gradient(gradf)[0]

        delta_feature = np.stack([sample_feature, gradf, grad2f])
        delta_filename = '{}-{}-delta.npy'.format(base_filename, fn_name)
        np.save(delta_filename, delta_feature)

def parse_phn(phn_filename):
    """
        Reads sen_id.phn file and returns a tuple of all the phoneme information
        Each phoneme will have the start and end extended evenly till it is bigger
        than MIN_DATAPOINTS. The rationale behind extending is susch that the model
        will have at least 8 frames of MFCC to make the decision

        Parameter
        ---------
            phn_filename: str: sen_id.phn filename
        Return
        ------
            phn_list: list: a list of tuples (phn_code, start, end)
    """
    phn_list = []

    with open(phn_filename, 'r') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            splt = line.split(' ')
            start, end, phoneme = int(splt[0]), int(splt[1]), splt[2]
            # remove the newline from phoneme
            phoneme = phoneme[:-1]
            # extend the start and end if file is too short
            diff = end - start
            if diff < MIN_DATAPOINTS:
                extension = MIN_DATAPOINTS - diff
                if start < (extension / 2):
                    end += extension + (extension - start)
                    start = 0
                elif i + 1 == len(lines):
                    start -= extension
                else:
                    start -= extension / 2
                    end += extension / 2

            phn_list.append((phoneme, start, end))

    return phn_list

def convert_wav_to_riff(wav_filename, riff_filename):
    """
        Converts wav files to riff files using linux package 'sox'

        Parameter
        ---------
            wav_filename: str: input wav filename
            riff_filename: str: output wav filename

        Return
        ------
            None
    """
    os.popen('sox {0} {1}'.format(wav_filename, riff_filename)).read()

def mkdir(dir):
    """
        Create the directory if it does not exist

        Parameter
        ---------
            dir: str: directory
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

def main():
    # create the data directory to store the features
    mkdir(DATA_DIR)

    for train_test in ["train", "test"]:
        train_test_dir = os.path.join(TIMIT_DIR, train_test)
        data_train_test_dir = os.path.join(DATA_DIR, train_test)
        mkdir(data_train_test_dir)

        for accent in next(os.walk(train_test_dir))[1]:
            acc_dir = os.path.join(train_test_dir, accent)
            data_acc_dir = os.path.join(data_train_test_dir, accent)
            mkdir(data_acc_dir)

            for speaker in next(os.walk(acc_dir))[1]:
                print("Processing speaker: {}".format(speaker))

                spk_dir = os.path.join(acc_dir, speaker)
                data_spk_dir = os.path.join(data_acc_dir, speaker)
                mkdir(data_spk_dir)

                regex = os.path.join(spk_dir, '*.phn')
                phn_file_list = glob.glob(regex)

                for phn_file in phn_file_list:
                    full_sen_id = os.path.splitext(phn_file)[0]
                    sen_id = os.path.basename(full_sen_id)
                    data_sen_dir = os.path.join(data_spk_dir, sen_id)
                    mkdir(data_sen_dir)

                    # check if riff files exist. if not, create riff files
                    riff_filename = full_sen_id + '.riff.wav'
                    if not os.path.exists(riff_filename):
                        convert_wav_to_riff(full_sen_id + '.wav', riff_filename)

                    phn_list = parse_phn(phn_file)

                    for phn, start, end in phn_list:
                        whole = wave.open(riff_filename)
                        whole.readframes(start)
                        frames = whole.readframes(end - start)

                        # create a directory for each phoneme
                        data_phn_dir = os.path.join(data_sen_dir, phn)
                        mkdir(data_phn_dir)

                        # create the phoneme filename
                        base_path = os.path.join(data_phn_dir,
                            '{sen_id}-{start}-{end}'.format(
                            sen_id=sen_id, start=start, end=end))
                        phn_wav_filename = '{}.riff.wav'.format(base_path)

                        out = wave.open(phn_wav_filename, 'w')
                        out.setparams(whole.getparams())
                        out.writeframes(frames)
                        out.close()

                        extract_features(phn_wav_filename, base_path)

                        if os.path.exists(phn_wav_filename):
                             os.remove(phn_wav_filename)

if __name__ == '__main__':
    main()