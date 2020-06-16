#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */home/workspace/ImageClassifier/predict.py
#
# PROGRAMMER: Hesham Mohamed Rmah.
# DATE CREATED: 09/27/2019
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --dir <directory with images> --topk <Number of top K matches>
#             --category_names <Mapping from categories to real names.>
#             --gpu <Type of the Device is GPU>
#   Example call:
#    python predict.py --dir flowers/test/30/image_03482 --topk 5 category_names 'cat_to_name.json' --gpu
##

# Imports python modules
import argparse
import json
import torch
import numpy as np

from math import ceil
from PIL import Image
from train import check_gpu
from torchvision import models


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
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--dir', type=str, default='flowers/test/30/image_03488.jpg',
                        help='Impage File Path.')
    parser.add_argument('--checkpoint', type=str, default ='my_checkpoint.pth',
                        help='Checkpoint file Path.')
    parser.add_argument('--topk', type=int, default=5,
                        help='Choose top K matches.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Mapping from categories to real names.')
    parser.add_argument('--gpu', action='store_true', 
                        help='Type of the Device is GPU')
    # returns parsed argument collection
    return parser.parse_args()


def load_checkpoint(checkpoint_Path):
    '''
    loads a checkpoint and rebuilds the model
    '''
    global mode
    checkpoint = torch.load(checkpoint_Path)
    # default architecture is 'vgg16
    if checkpoint['architecture'] == 'vgg16':
        mode = models.vgg16(pretrained=True)
        mode.name = "vgg16"
    elif checkpoint['architecture'] =='alexnet':
        mode = models.alexnet(pretrained=True)
        mode.name = 'alexnet'
    elif checkpoint['architecture'] =='densenet121':
        mode = models.densenet121(pretrained=True)
        mode.name = 'densenet121'
    else:
        print('Undefined Architecture.')
    # Freeze parameters so we don't backprop through them
    for param in mode.parameters():
        param.requires_grad = False

    mode.class_to_idx = checkpoint['class_to_idx']
    mode.classifier = checkpoint['classifier']
    mode.load_state_dict(checkpoint['state_dict'])

    return mode

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    test_image = Image.open(image)
    # Get dimensions
    width , height = test_image.size
    # Find shorter size and create settings to crop shortest side to 256
    if width < height:
        aspect_ratio = height / width
        new_size = [256 ,256 * aspect_ratio]
        center = new_size[0]/2 , new_size[1]/2
    else:
        aspect_ratio = width / height
        new_size = [256 * aspect_ratio ,256]
        center = new_size[0]/2 , new_size[1]/2
    test_image.thumbnail(size = new_size)
    # Find pixels to crop on to create 224x224 image
    top ,bottom = center[1]-(224/2) ,center[1]+(224/2)
    right ,left = center[0]+(224/2) ,center[0]-(224/2)
    test_image = test_image.crop((left, top, right, bottom))
    np_image = np.array(test_image)/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2 ,0 ,1)
    
    return np_image


def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    # Set model to evaluate
    model.eval()
    # Convert image from numpy to torch
    tor_image = torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FloatTensor).to(device)
    log_probs = model.forward(tor_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(topk)
  
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    for i, j in enumerate(zip(flowers, probs)):
        print("Rank {}:".format(i+1),
              "Flower: {}, Probability: {}%".format(j[1], ceil(j[0]*100)))

        
# Main program function defined below
def main():
    """
    defining the main function
    """
    # Creates & retrieves Command Line Arugments
    args = arg_parser()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    device = check_gpu(gpu_arg=args.gpu)

    print("Type of the Device is :{}".format(device))
    
    top_probs, top_labels, top_flowers = predict(args.dir, model, args.topk, cat_to_name, device)

    print_probability(top_flowers, top_probs)


# Call to main function to run the program
if __name__ == '__main__':
    main()
