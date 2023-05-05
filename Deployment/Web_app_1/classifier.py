import numpy as np
import time
from sklearn.model_selection import train_test_split
#from tqdm.autonotebook import tqdm, trange

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import cv2


BATCH_SIZE = 32
import torch._utils
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
# from model import CNNModel
# from datasets import train_loader, valid_loader
# from utils import save_model, save_plots

import torchvision.models as models
import torch._utils
def predictor(filename):
    transform_inf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(300),
        transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.5, 0.5, 0.5],
    #         std=[0.5, 0.5, 0.5]
    #     )
    ])  



    # the computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # list containing all the class labels
    labels = [
        'Knob', 'No knob', 
        ]


    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    model_loaded = resnet18
    model_loaded.load_state_dict(torch.load(r"C:\Users\vikas\OneDrive\Desktop\Work\Object_detector\Classifier\classifier_01.pth"))
    model_loaded.eval()
    model_loaded.to(device)


    image = cv2.imread(filename)
    # get the ground truth class
    # gt_class = args['input'].split('/')[-2]
    orig_image = image.copy()
    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform_inf(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model_loaded(image.to(device))
    output_label = torch.topk(outputs,1)
    pred_class = labels[int(output_label.indices)]

    return pred_class

# pred_class = predictor(r"C:\Users\vikas\OneDrive\Desktop\Work\Object_detector\Classifier\Knob\IMG_4183.jpg")
# print(pred_class)
