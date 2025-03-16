import numpy as np
import torch
import torchvision
import torch.utils.tensorboard as tb

import numpy as np

def train():
    # this has to do two things: 
    ## depth estimation 
    ## road semantic segmentation, which classifies each as left boundary, right boundary, or not boundary

    # depth estimation

    