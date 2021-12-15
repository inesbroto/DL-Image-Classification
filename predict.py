
import argparse
import os
import PIL

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils_eff import efficientnet
from models.efficient import EfficientNet

def predict(model_path, img_path):

    device = torch.device('cpu')
    blocks_args, global_params = efficientnet()
    model = EfficientNet(blocks_args, global_params).to(device)
    

    #load model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # load the image and prepare it to feed the model
    image = Image.open(img_path)
    data = np.asarray(image)
    img_resized = image.resize((64,64))
    
    tfms = transforms.Compose([transforms.Resize((64,64)), transforms.CenterCrop((64,64)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(image).unsqueeze(0)
    


    #make prediction
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True) 
    
    #load the labels
    with open('./data/de_words.txt') as f:
        lines = f.readlines()

    print(pred, output[pred], lines[pred])
