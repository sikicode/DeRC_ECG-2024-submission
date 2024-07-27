#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################
# Helper 
import torch
from torch import nn
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms import functional as TF
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import re

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2        
import os
import pandas as pd
import glob
from tqdm import tqdm, trange
import subprocess
import requests

# get_file_names
def get_file_names(input_files: list) -> list:
    input_file_names = []
    for input_file in input_files:
        temp_file_name = input_file.split('/')[-1].strip('.png')
        input_file_names.append(temp_file_name)
    print ('Total', len(input_file_names), 'files')
    return input_file_names


# Conv backbone
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#UNet model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)
        self.double_conv1 = DoubleConv(64, 128)
        self.double_conv2 = DoubleConv(128, 256)
        self.double_conv3 = DoubleConv(256, 512)
        self.double_conv4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv_up1 = DoubleConv(1024, 512)
        self.double_conv_up2 = DoubleConv(512, 256)
        self.double_conv_up3 = DoubleConv(256, 128)
        self.double_conv_up4 = DoubleConv(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.double_conv1(x2)
        x3 = self.down2(x2)
        x3 = self.double_conv2(x3)
        x4 = self.down3(x3)
        x4 = self.double_conv3(x4)
        x5 = self.down4(x4)
        x5 = self.double_conv4(x5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.double_conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv_up3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv_up4(x)
        logits = self.outc(x)
        logits = self.sigmoid(logits)
        return logits

#Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

#Focal Loss https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss
class FocalLoss(nn.Module): 
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.focal_loss(inputs, targets)


def calculate_dice(target, prediction):
    """
    Calculate Dice Coefficient for a single image.
    
    Args:
        target (np.array): The ground truth binary segmentation image.
        prediction (np.array): The predicted binary segmentation image.
    
    Returns:
        float: The Dice Coefficient.
    """
    intersection = np.logical_and(target, prediction)
    dice_score = 2 * np.sum(intersection) / (np.sum(target) + np.sum(prediction))
    return dice_score

def load_model_layers(model, directory="model_layers"):
    for name, param in model.named_parameters():
        file_path = os.path.join(directory, f"{name.replace('.', '_')}.pt")
        if os.path.exists(file_path):
            param_data = torch.load(file_path, map_location=torch.device('cpu'))
            param.data = param_data

def get_classification_model():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = model.conv1.weight.data.mean(dim=1, keepdim=True)

    # Modify the fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # Adjusting for binary classification
    load_model_layers(model, directory='/challenge/pretrain_model/ResNet18/')
    return model
    
def get_denoise_model():
    model = UNet(n_channels=3, n_classes=1)
    load_model_layers(model, directory='/challenge/pretrain_model/UNet/')
    return model

# End of Helper

# Weijie_model

# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjusted for a single channel
])

noise_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization values
])

class ECGNoisedImageDataset(Dataset):
    def __init__(self, image_path_list, transform = None, 
                 noise_transform = None, denoise_model = None, device = None):
        """
        Args:
            dataframe (pd.DataFrame): A pandas dataframe containing 'file_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_path_list = image_path_list
        # self.label_list = label_list
        self.noise_transform = noise_transform
        self.transform = transform
        self.denoise_model = denoise_model
        self.device = device

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        
        input_image = cv2.imread(self.image_path_list[idx], cv2.IMREAD_COLOR)
        input_image = self.noise_transform(input_image)        
        input_image = TF.resize(input_image, (896, 1152))
        input_image = torch.sigmoid(input_image).to(device)
        input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        output_image = self.denoise_model(input_image)

        if self.transform:
            output_image = self.transform(output_image)

        # label = self.label_list[idx]
        return output_image #, torch.tensor([label, ~label]).float(), self.image_path_list[idx]

def get_current_directory():
    return os.path.dirname(os.path.abspath(__file__))

def load_data_weijie(data_path):
    
    current_directory = get_current_directory()    
    # PATH = '/home/weijiesun/physinet2024/model/UNet_torch.model' #./model/UNet_torch.model'
    # Weijie Need to update
    denoise_model = get_denoise_model().to(device)
    # denoise_model = torch.load(path_unet, map_location ='cpu').to(device)
    denoise_model.eval()
    
    # PATH = '/home/weijiesun/physinet2024/model/ResNet_C_torch.model' #'./model/ResNet_C_torch.model'
    model = get_classification_model().to(device)
    model.eval()

    # pattern = re.compile(r'.*\.(png|jpg|jpeg)$')
    # temp_list = [f for f in os.listdir(data_path) if pattern.match(f)]
    # temp_path = data_path + '-0.png'
    temp_list = [data_path]
    # temp_list = [temp_list[0]]
    # print ('Weijie: data_path',data_path)
    # print(f'temp_list - new: datapath: {data_path} -- temp_list: {temp_list}')
    # print ('Weijie: feel there is a problem', len(temp_list), data_path)

    batch_size = 1
    # temp_list = temp_list[:1]
    test_dataset = ECGNoisedImageDataset(image_path_list=temp_list, transform = transformations, 
                                         noise_transform = noise_transform, denoise_model = denoise_model, device = device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    prob_list = []
    target_list = []
    pred_binary_list = []
    with tqdm(total=len(test_dataloader)) as pbar:
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            inputs = inputs.reshape(1, 1, 224, 224)
            outputs = model(inputs)
            print(outputs)
            _, pred_binary = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            # targets = targets[:, 1]
            prob_list.extend(probs.detach().cpu().numpy())
            # target_list.extend(targets.detach().cpu().numpy())
            pred_binary_list.extend(pred_binary.detach().cpu().numpy())
            pbar.update(1)
    # print(f'pred_binary_list: {pred_binary_list}')
    modified_list = ['Abnormal' if x == 1 else 'Normal' for x in pred_binary_list]
    # print(f'pred_binary_list: {modified_list}')
    
    return modified_list

# Train your digitization and classification models.
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    # print(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # digitization_features = list()
    # classification_features = list()
    # classification_labels = list()

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        # record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        # features = extract_features(record)
        
        # digitization_features.append(features)

        # Some images may not be labeled...
        # labels = load_labels(record)
        # if any(label for label in labels):
            # classification_features.append(features)
            # classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    # if not classification_labels:
        # raise Exception('There are no labels for the data.')

    # Train the models.
    # if verbose:
        # print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    # digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    
    # This very simple model trains a random forest model with these very simple features.
    # classification_features = np.vstack(classification_features)
    # classes = sorted(set.union(*map(set, classification_labels)))
    # classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    # n_estimators   = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    # classification_model = RandomForestClassifier(
    #    n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)

    digitization_filename = os.path.join("/challenge/pretrain_model/", 'digitization_model.sav')
    # print(digitization_filename)
    digitization_model = joblib.load(digitization_filename)
    d = digitization_model
    filename = os.path.join(model_folder, 'digitization_model.sav')
    # print(filename)
    joblib.dump(d, filename, protocol=0)
    
    # Save the models.
    # save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        # print ("Weijie - We should not train the data, we use the pretrain data")
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)
    # print(device)
    classification_model = get_classification_model().to(device)
    classification_model.eval()
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.

    # Load the digitization model.
    model = digitization_model['model']
    # print(model)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # print(record)
    # Extract the features.
    # try:
    #     features = extract_features(record + "-0.png")
    # except RuntimeError as e:
    #     features = extract_features(record)
    # features = features.reshape(1, -1)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model))
    signal = np.random.default_rng(seed=seed).uniform(low=-10, high=10, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.float32)
    
    # Run the classification model; if you did not train this model, then you can set labels = None.
    # Split the path by '/'
    # parts = path.split('/')
    
    # Remove the last portion
    # new_path = '/'.join(parts[:-1])
    # print(f'run_classification_model -> data_record: {record}')
    # print(f'record: {record} -- new_path: {new_path}')
    
    try:
        labels = load_data_weijie(record + "-0.png")
    except:
        labels = load_data_weijie(record)  
    print(labels)
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    print ('Weijie - Should not use this extract_features function')
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    print ('Weijie - Should not save the train models')
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)
