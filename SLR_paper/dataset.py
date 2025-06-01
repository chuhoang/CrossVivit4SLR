import numpy as np
import timm
# import decord
import torch
from decord import cpu, gpu
from torch.utils.data import DataLoader
from decord import VideoReader, VideoLoader
from torchvision.datasets import DatasetFolder
from fvcore.common.checkpoint import Checkpointer
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, Grayscale, ToPILImage
from torchvision.transforms.v2 import RandomHorizontalFlip
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
import os
def LoadPath_flatten_embedding(path):
    # Gói block với size (16, 320, 240, 3)
    vl = VideoLoader([path], ctx=[cpu(0)], shape=(32, 480 ,720 , 3), interval=0, skip=1, shuffle=0)
    temp_array = []
    for batch in vl:
        temp_array.append(torch.from_numpy(batch[0].asnumpy()))
    return torch.stack(temp_array)
        
def LoadPath(path):
    # k = path.split('/')[3]
    # mode = path.split('/')[5]
    # if mode == 'val':
    #     new_path = path.replace('dataset', 'SL-PTIT-50').replace('val/', '')
    # else:
    #     new_path = path.replace('dataset', 'SL-PTIT-50').replace('train/', '')
    # latest_path  = new_path.replace(k, 'trunghm')
    
    # if not os.path.exists(latest_path):
    # latest_path = latest_path.replace('avi', 'mp4')
    vr = VideoReader(path, ctx=cpu(0), height=224, width=224)
    data_video = vr.get_batch([np.arange(0, len(vr), 2)]).asnumpy()
    data_video = torch.tensor(data_video)
    T, H, W, C = data_video.shape
    if T <= 32:
        padding_video = torch.zeros(32 - T, H, W, C)
        data_video = torch.cat((data_video, padding_video), dim = 0)
    else:
        data_video = vr.get_batch([np.arange(0, len(vr), 4)]).asnumpy()
        data_video = torch.tensor(data_video)
        T, H, W, C = data_video.shape
        if T > 32:
            data_video = data_video[0:32, :, :, :]
            T = 32
        padding_video = torch.zeros(32 - T, H, W, C)
        data_video = torch.cat((data_video, padding_video), dim = 0)
    return data_video
def checkFile(path):
    if path.endswith(('.mp4', '.avi', '.webm')):
        return True

# datasets = LoadDatasets()
def transform_video(datasets):
    # count = 0
    transform_video = Compose([
        RandomHorizontalFlip(p=0.5)
    ])
    datasets = transform_video(datasets)
    check = []
    for data in datasets:
        # count+=1
        k = data.numpy()
        # k/=255
        # k = k.astype(np.uint8)
        k/=255
        transform = Compose([
            ToTensor(),
            Normalize(mean = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
            
            # Grayscale(3),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        k = transform(k)          
            # save_image(k, '/home/signlanguage/jupyter_workspace/sign_language_test1/output/image{count}.png'.format(count = str(count)))
        check.append(k)
    results = torch.stack(check)
    return results.to('cuda')
            
def transform_video_val(datasets):
    # count = 0
    check = []
    for data in datasets:
        # count+=1
        k = data.numpy()
        # k/=255
        # k = k.astype(np.uint8)
        k/=255
        transform = Compose([
            ToTensor(),
            Normalize(mean = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
            
            # Grayscale(3),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        k = transform(k)          
            # save_image(k, '/home/signlanguage/jupyter_workspace/sign_language_test1/output/image{count}.png'.format(count = str(count)))
        check.append(k)
    results = torch.stack(check)
    return results.to('cuda')

def LoadDatasets(path, transform_video):
    datasets = DatasetFolder(root = path, loader = LoadPath, transform=transform_video, is_valid_file=checkFile)
    return datasets