import sys
import os
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import argparse
from text_to_class.dataloading import TextLoader
import torch
import torch.nn as nn
from text_to_class.models import LSTM
from metrics import eval
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser(description='Testing text to class.....')
parser.add_argument('--path', default = 'data/action_classes.txt', type = str,
                            help= 'Set the relative path to find the file that contains the dataset.')
parser.add_argument('--sequence_length', default = 30, type = int,
                            help= 'Set the maximum length for each item that will be given to the model.')
parser.add_argument('--text_path', type=str, default='text_to_class/LSTM-checkpoint-3700',
                     help='set path (prefix name) to load state for text to class')
parser.add_argument('--testing_path', default = 'data/testing_data.txt', type = str,
                            help= 'Set the relative path to find the file that contains the testing data.')

args = parser.parse_args()
path = args.path
sequence_len = args.sequence_length
text_path = args.text_path
testing_path = args.testing_path

def train(humanDescription, dataset, network):
 
    if torch.cuda.is_available():
        tensor              = torch.tensor(dataset.prepareTxtForTensor(humanDescription )).cuda().unsqueeze_(0)
    else: 
        tensor              = torch.tensor(dataset.prepareTxtForTensor(humanDescription )).unsqueeze_(0)

    output              = network(tensor)
    probability, action = output.max(1)
    actionClassName      = dataset.getClassNameFromIndex(action  + 1)

    return (actionClassName, probability)

def predict(test_description, dataset, network):
    class_predict = []
    probs = []
    for des in test_description:
        label, prob = train(des, dataset, network)
        class_predict.append(label)
        probs.append(prob)

    return (class_predict, probs)

if __name__ == "__main__":

    # Load LSTM model to get the category predicted from natural language
    rnnType     = nn.LSTM
    rnnSize     = 512
    embedSize   = 512
    itemLength  = 30
    loadEpoch   = 3700

    current_path = os.getcwd()
    dataset_path = os.path.join(current_path, 'text_to_class', 'data', 'action_classes.txt')
    test_path = os.path.join(current_path, 'text_to_class', testing_path)

    if not dataset_path:
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    dataset = TextLoader(dataset_path, item_length = itemLength)
    vocal_size = len(dataset.vocabulary) 

    network = LSTM(rnnType, rnnSize, embedSize, vocal_size, ngpu=1)
    network.loadState(os.path.join(current_path, text_path))

    testing_data = pd.read_csv(test_path, sep = "\t", header=None)
    test_classes = list(testing_data.iloc[:, 0])
    test_des = testing_data.iloc[:, 1]
    
    predict_classes, predict_probs = predict(test_des, dataset, network)

    # Combine true and predicted labels to fit the encoders properly
    all_labels = np.unique(test_classes + predict_classes)

   # Encode labels to numeric values
    le = LabelEncoder()
    le.fit(all_labels)
    true_labels_encoded = le.transform(test_classes)
    predicted_labels_encoded = le.transform(predict_classes)

    '''
    # Use for ROC_AUC
    # One-hot encode labels
    ohe = OneHotEncoder(sparse_output=False)
    all_labels_encoded = le.transform(all_labels).reshape(-1, 1)
    ohe.fit(all_labels_encoded)

    true_labels_ohe = ohe.transform(true_labels_encoded.reshape(-1, 1))
    predicted_labels_ohe = ohe.transform(predicted_labels_encoded.reshape(-1, 1))
    '''

    print("Evaluate the model with the label: ", all_labels[0])
    eval(true_labels_encoded , predicted_labels_encoded, class_index=0)


