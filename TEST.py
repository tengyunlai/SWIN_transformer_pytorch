import torch
import torch.nn as nn
import cv2
from models import swin_transformer
from models import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(net)
pre_state_dict = torch.load('/root/small_section/Image-train-Swin-transformer-main/output/swin_base_patch4_window12_384/default/ckpt_epoch_110.pth',map_location='cpu')

pre_state_dict['config'].defrost()
pre_state_dict['config'].MODEL.NUM_CLASSES  = 100
pre_state_dict['config'].freeze()
print(pre_state_dict['config'])
net = build_model(pre_state_dict['config'])

net.load_state_dict(pre_state_dict['model'])
net.to(device)

