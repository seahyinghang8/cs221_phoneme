# CS221 Project - Real-time Phoneme Classification for Learning Pronuciation

### Getting Started

The project is built using python2.7.
You must first install all the requirement packages. Create a virtualenv (optional) and install the required packages by running 
``` pip install -r requirements.txt ```
Since different OS will need different versions of pytorch, please install pytorch and torchvision as according to the instructions on pytorch's main page.

### Dataset
In this project, we are using Timit dataset. The dataset should be placed in the project root directory under the folder name 'timit'.
In our submission, we provided a small subset of the dataset that is available to the public from NLTK.

### Data Preprocessing
We will first preprocess the data to extract the features. The voice clip will be chopped into individual phonemes based on the \*.phn file. We will then apply mfcc / logfbank and deltas to each voice clip.

For the preprocessing code to work, you must first install 'sox' by running the following:
``` apt-get install sox ```

To start the preprocessing code to perform feature extraction, run:
``` python preprocess.py ```

Since we normalize the input before passing it into the model, we can calculate the mean and std for each feature by running the following:
``` python utils/computer_mean_std.py --type <mfcc/mfcc-delta/logfbank_40/logfbank_40-delta> ```
The type can be any of the feature type listed in the command.

In our implementation of the dataloader, we map each phoneme's english label to an integer label using a mapping located in the folder phn_index_maps/. As we constrained our problem by reducing the set of phonemes for classification, our experiments will be running based on `phn_index_map_constrained.yaml`. phn_index_maps/process_index.py is used to create these yaml mapping files from the `phn_index_maps/raw_map.yaml`.

To visualize the extracted features to have a sense of the data, you can run the following:
``` python utils/feature_visualization.py --type <mfcc/mfcc-delta/logfbank_40/logfbank_40-delta> ```
This will save a plot of 12 examples for each phoneme into a folder in the root directory `<mfcc/mfcc-delta/logfbank_40/logfbank_40-delta>-features-plot/`.

### Training the Model

Our models are stored in the directory `models`. All the training configs are stored in the directory `configs/`. Here is a table of the configs and their respective descriptions.

Config Name                   | Description
----------------------------- | ---------------------------------------------------------------------------------------------
constrained.yaml              | Constrained experiment with fixed hyperparameters for 2 layer CNN that gives best results
constrained_lr.yaml           | Constrained experiment with a range of hyperparamers with Linear Regression using logfbank
constrained_cnn_2.yaml        | Constrained experiment with a range of hyperparamers with 2 layer CNN using logfbank
constrained_cnn_5.yaml        | Constrained experiment with a range of hyperparamers with 5 layer CNN using logfbank
constrained_cnn_10.yaml       | Constrained experiment with a range of hyperparamers with 10 layer CNN using logfbank
constrained_cnn_2_mfcc.yaml   | Constrained experiment with a range of hyperparamers with 2 layer CNN using mfcc
constrained_cnn_2_delta.yaml  | Constrained experiment with a range of hyperparamers with 2 layer CNN using logfbank + delta

To train our model, run the following:
``` python train.py --config <config_filename> ```

To obtain the results as mentioned in our paper, use the constrained_lr.yaml configuration.
In order to obtain the best set of hyperparameters to train the network, we performed hyperparameter sweeping for the learning rate, l2 regularization and dropout. All the experimental logs are stored in the directory `expts/` automatically.

All the experiments that we ran to obtain our results are recorded in `expts/`. Only the sweeps with the best performing model is saved.

To make it easier to extract information from the log, run the following:
``` python utils/log_to_csv.py ```
This will extract the important information of the sweeps into a `sweeps.csv` file in the experiment directory where the log is stored.

### Evaluating the Model

To evaluate the performance of the model on the testing set, run the following command:
``` python eval.py --config <config_filename> --load <path_to_model> ```
Evaluation will also automatically plot the confusion matrix at the end for error analysis.


### Running the Demo

We have created a simple demo using pyaudio with our best performing model. Audio input captured from the microphone is continuously being processed in the background to generate the chart.

To start the demo, run the following command:
``` python demo.py ```

If you are running the model on MacOS, you can try to install 'pyaudio' executing commands the following in order

~~~~
brew update 
brew install portaudio
brew link --overwrite portaudio
$ pip install pyaudio
~~~~