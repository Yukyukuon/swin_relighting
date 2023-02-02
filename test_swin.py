import random
import os
import torch
torch.autograd.set_detect_anomaly(True)
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import argparse

import cv2
from torch.autograd import Variable

from models.swin_transformer import SwinTransformerUnet
from utils.utils_SH import *

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------

def strip_prefix(state_dict, prefix='module.'):
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped_state_dict = {}
    for key in list(state_dict.keys()):
        stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
    return stripped_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = 'output/SwinTransformer_model_5.pth'

model = SwinTransformerUnet().to(device)
model.load_state_dict(strip_prefix(torch.load(ckpt_path)))
model.eval()

criterion = torch.nn.MSELoss().to(device)

test_data_path = 'data/test_dataset'
hq_path = 'data/DPR_dataset'
saveFolder = 'result_Swin'

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

def read_original_img(path):
    img = cv2.imread(path)
    row, col, _ = img.shape
    img = cv2.resize(img, (512, 512))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())
    return inputL, Lab, row, col

for file in os.listdir(test_data_path):

    inputL, Lab, row, col = read_original_img(os.path.join(test_data_path, file))

    img_id = int(file.split('.')[0]) - 1
    hq_img_path = 'imgHQ0000{}'.format(img_id)

    for i in range(5):
        light_path = os.path.join(hq_path, hq_img_path, '{}_light_0{}.txt'.format(hq_img_path,i))

        sh = np.loadtxt(light_path)
        sh = sh[0:9]
        sh = sh * 0.7

        #--------------------------------------------------
        # rendering half-sphere
        sh = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        cv2.imwrite(os.path.join(saveFolder, \
                '{}_light_{:02d}.png'.format(hq_img_path, i)), shading)
        #--------------------------------------------------

        #----------------------------------------------
        #  rendering images using the network
        sh = np.reshape(sh, (1,9,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, outputSH  = model(inputL, sh)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        cv2.imwrite(os.path.join(saveFolder, \
            '{}_{:02d}.jpg'.format(hq_img_path, i)), resultLab)
        

    
    




