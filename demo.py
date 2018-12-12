from __future__ import print_function

import pyaudio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from python_speech_features import logfbank
from configs.config_parser import parse
import torch
from models.cnn import cnn2Layer
import torchvision.transforms as transforms
import math
import os

CONFIG = "constrained_cnn_2"
MODEL_PATH = "weights/best_model.pth"

# Load model
class Config:
    def __init__(self):
        self.config = CONFIG

keep_going = True
plot_feature = np.zeros((40, 100))
plot_values = np.zeros((24, 15))
# plot_values = np.zeros((12, 15))


def main():
    global keep_going

    cfg = parse(Config())

    model = cnn2Layer(1, 10, 17, 40)
    # model = cnn2Layer(1, 10, 8, 40)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


    FORMAT = pyaudio.paInt16 # We use 16 bit format per sample
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096 # num data read from the buffer, aim to get 2 logfbankframes

    audio = pyaudio.PyAudio()

    # Claim the microphone
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE, 
                        input=True,
                        frames_per_buffer=CHUNK)

    WIN_LENGTH = 0.02
    WIN_STEP = 0.005

    f, axarr = plt.subplots(1,2)

    def plot_data(in_data):
        global plot_feature
        global plot_values
        # get and convert the data to float
        audio_data = np.fromstring(in_data, np.int16)
        extracted_feature = logfbank(audio_data, RATE, winlen=WIN_LENGTH, winstep=WIN_STEP, nfilt=40)
        print(extracted_feature.shape)
        num_frame = extracted_feature.shape[0] - 2
        plot_feature = np.roll(plot_feature, -num_frame, axis=1)
        plot_feature[:, -num_frame:] = extracted_feature.swapaxes(0, 1)[:, 2:]

        feature_t = torch.from_numpy(extracted_feature[3:13]).type(torch.FloatTensor)
        old_shape = feature_t.shape
        feature_t = feature_t.reshape((1, old_shape[0], old_shape[1]))
        normalize = transforms.Normalize(mean=[7.7120], std=[3.7074])
        feature = normalize(feature_t)
        feature_t = feature_t.unsqueeze(0)
        output = (model(feature_t))[0].detach().numpy()
        plot_values = np.roll(plot_values, -1, axis=1)
        plot_values[:, -1] = [math.exp(output[i]) for phn, i in cfg.phn_idx_map.items()]
        os.system('clear')
        classes = {phn: math.exp(output[i]) for phn, i in cfg.phn_idx_map.items()}
        for cls in cfg.phn_idx_map:
          print(cls, classes[cls])
        print(max([(val, phn) for phn, val in classes.items()]))

        axarr[0].cla()
        axarr[1].cla()

        axarr[0].imshow(plot_feature, cmap="Greys", vmin=0, vmax=15)
        axarr[1].imshow(plot_values, cmap="Greys", vmin=0, vmax=1)

        del extracted_feature
        # Show the updated plot, but without blocking
        plt.pause(0.001)

        if keep_going:
            return True
        else:
            return False

    # Open the connection and start streaming the data
    stream.start_stream()
    print("\n+---------------------------------+")
    print("| Press Ctrl+C to Break Recording |")
    print( "+---------------------------------+\n")

    plt.ion()
    # Loop so program doesn't end while the stream is open
    while keep_going:
        try:
            plot_data(stream.read(CHUNK))
        except KeyboardInterrupt:
            keep_going=False
        except:
            pass

    plt.ioff()
    plt.show()

    # Close the audio streaming file
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == '__main__':
    main()