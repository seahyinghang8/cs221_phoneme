# Parse the log to get the loss, error rate etc. for all the experiments and store the data as a CSV
import os
import argparse
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def processLine(line):
    # Identify if the line of log contains important information
    if 'Sweep Count' in line:
        return 'swp_count'
    if 'Config' in line:
        return 'c_start'
    if '\n' == line:
        return 'c_end'
    if ' Epoch: ' in line:
        return 'data'
    if ' Test PER: ' in line:
        return 'test_err'

def getStringBetween(line, start_str, end_str):
    # Extract the substring between the start and end string
    start_index = line.find(start_str) + len(start_str)
    end_index = line.find(end_str)
    return line[start_index: end_index]

def extractTestErr(line):
    return float(getStringBetween(line, 'PER: ', '%'))

def extractTrainLoss(line):
    return float(getStringBetween(line, 'Train Loss: ', '\tTrain PER'))

def extractValLoss(line):
    return float(getStringBetween(line, 'Val Loss: ', '\tVal PER'))

def extractTrainErr(line):
    return float(getStringBetween(line, 'Train PER: ', '\tVal Loss'))

def extractValErr(line):
    return float(getStringBetween(line, 'Val PER: ', '%\n'))

def readLog(file):
    # Reads the log to extract all the important information
    expts = []
    curr_expt = None
    lowest_val_err = 100
    in_config = False

    for line in file:
        out = processLine(line)

        if not out and not in_config: continue
        if out == 'swp_count':
            # marks the start of an experiment sweep
            curr_expt = {}
            lowest_val_err = 100
            curr_expt['train_data'] = []
        elif out == 'test_err':
            # marks the end of an experiment sweep. record the test error
            curr_expt['test_err'] = extractTestErr(line)
            curr_expt['lowest_val_err'] = lowest_val_err
            #append to expt list
            expts.append(curr_expt)
        elif out == 'c_start':
            # marks the start of reading the config text
            in_config = True
            curr_expt['config_text'] = ''
        elif out == 'c_end':
            # marks the end of reading the config text
            in_config = False
        elif in_config:
            # record the text
            curr_expt['config_text'] += line
            if 'l2_regularizer' in line:
                curr_expt['l2_regularizer'] = float(getStringBetween(line, 'l2_regularizer: ', '\n'))
            elif 'dropout' in line:
                curr_expt['dropout'] = float(getStringBetween(line, 'dropout: ', '\n'))
            elif 'learning_rate' in line:
                curr_expt['learning_rate'] = float(getStringBetween(line, 'learning_rate: ', '\n'))
        elif out == 'data':
            curr_expt['train_data'].append({
                'train_loss': extractTrainLoss(line),
                'val_loss': extractValLoss(line),
                'train_err': extractTrainErr(line),
                'val_err': extractValErr(line)
            })
            lowest_val_err = min(lowest_val_err, extractValErr(line))
            
    return expts

def getXYZ(data, a_name, x_name, y_name, z_name):
    # extracts the best validation error and their respective hyperparameters
    a = []
    x = []
    y = []
    z = []

    for expt in data:
        a.append(expt[a_name])
        x.append(expt[x_name])
        y.append(expt[y_name])
        z.append(expt[z_name])

    return [np.asarray(ls) for ls in [a, x, y, z]]

def main():
    parser = argparse.ArgumentParser(description='Parse the log to get data for plotting')
    parser.add_argument('--file', '-f', type=str, help='name of the log file relative to the project root directory', required=True)
    args = parser.parse_args()

    with open(os.path.join(ROOT_DIR, args.file), 'r') as file:
        expt_list = readLog(file)

    a, x, y, z = getXYZ(expt_list, 'dropout', 'l2_regularizer', 'learning_rate', 'lowest_val_err')

    df = pd.DataFrame({
        'dropout': a,
        'l2_regularizer': x,
        'learning_rate': y,
        'lowest_val_err': z
        })

    df.to_csv(os.path.join(ROOT_DIR, os.path.dirname(args.file), 'sweeps.csv'))


if __name__ == '__main__':
    main()