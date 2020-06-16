#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/ImageClassifier/train.py
#
# PROGRAMMER: Hesham Mohamed Rmah.
# DATE CREATED: 09/27/2019
# PURPOSE: train image classifier model with traning set and validation set
#          save the traind model
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --data_dir <directory with images Folder> 
#             --arch <Models to use (vgg,densenet,alexnet)> --learning_rate <Set the learning rate>
#             --hidden_layers <Number of hidden layers> --epochs <number of epochs>
#             --print_every <Number of print every>
#             --gpu <Type of the Device is GPU>
#   Example call:
#    python train.py --data_dir flowers --arch 'vgg16' --learning_rate 0.001 --hidden_layers 4096 --epochs 5 --print_every 40 --gpu
##

# Imports python modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import argparse
import json
import torch
import os

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

# Functions defined below
def arg_parser():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse
    parser = argparse.ArgumentParser(description='Neural Network Settings')
    parser.add_argument('--data_dir', type=str,
                        default='flowers', help='Data Directory')
    parser.add_argument('--arch', type=str, default = 'densenet',
                        help='Models to use (vgg,densenet,alexnet)')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning Rate')
    parser.add_argument('--hidden_layers', type=int, default=4096,
                        help='Number of Hidden Layers')
    parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs')
    parser.add_argument('--print_every', type=int, default=40, help='Number of print every')
    parser.add_argument('--gpu', action='store_true', help='Type of the Device is GPU')
    # returns parsed argument collection
    return parser.parse_args()


def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    # Device agnostic code, automatically uses CUDA if it's enabled
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == "cpu":
        print("the device is set to CPU .")
    return device


def model(architecture="vgg16"):
    global mode
    if architecture == 'vgg':
        mode = models.vgg16(pretrained=True)
        mode.name = "vgg16"
        input_features = 25088
        print("Network architecture specified as vgg16.")
    elif architecture =='alexnet':
        mode = models.alexnet(pretrained=True)
        mode.name = 'alexnet'
        input_features = 9216
        print("Network architecture specified as alexnet.")
    elif architecture =='densenet':
        mode = models.densenet121(pretrained=True)
        mode.name = 'densenet121'
        input_features = 1024
        print("Network architecture specified as densenet121.")
    else:
        print('Undefined Architecture, Network architecture specified as vgg16. ')
        mode = models.vgg16(pretrained=True)
        mode.name = "vgg16"
        input_features = 25088
    # Freeze parameters so we don't backprop through them
    for param in mode.parameters():
        param.requires_grad = False
    return mode , input_features


def classifier(model, train_dir, input_features, hidden_Layers):
    num_classes = sum([len(folder) for r, folder, d in os.walk(train_dir)])
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_Layers, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_Layers, num_classes, bias=True)),
        ('output', nn.LogSoftmax(dim=1))]))
    return classifier


def deep_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device):
    model.to(device)
    steps = 0
    for e in range(epochs):
        running_loss = 0
        model.train()

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation_set(
                        model, validloader, criterion, device)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.3f} | ".format(
                          running_loss/print_every),
                      "Validation Loss: {:.3f} | ".format(
                          valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()

    print("Done Training !!!!")
    return model


def validation_set(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(validloader):

        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def check_accuracy(model, testloader, device):
    correct = 0
    total_all = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            N_, prediction = torch.max(outputs.data, 1)
            total_all += labels.size(0)
            correct += (prediction == labels).sum().item()

    print('Accuracy of the Network : %d%%' % ((correct / total_all) * 100))


def checkpoint(model, training_data):
    model.class_to_idx = training_data.class_to_idx
    checkpoint = {'architecture': model.name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Checkpoint Completed !.")


# Main program function defined below
def main():
    """
    defining the main function
    """
    # Creates & retrieves Command Line Arugments
    args = arg_parser()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(
                                                  224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(
        train_dir, transform=training_transforms)
    validation_datasets = datasets.ImageFolder(
        valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(
        test_dir, transform=testing_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        training_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        validation_datasets, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testing_datasets, batch_size=32, shuffle=False)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model_, input_features = model(architecture=args.arch)

    model_.classifier = classifier(model_, train_dir, input_features, hidden_Layers=args.hidden_layers)

    device = check_gpu(gpu_arg=args.gpu)
    
    print("Type of the Device is :{}".format(device))
    
    model_.to(device)
    # Define loss (criterion) and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_.classifier.parameters(),
                           lr=args.learning_rate)

    trained_model = deep_model(model_, trainloader, validloader,
                               args.epochs, args.print_every, criterion, optimizer, device)

    check_accuracy(model_, testloader ,device)
    checkpoint(trained_model, training_datasets)


# Call to main function to run the program
if __name__ == '__main__':
    main()
