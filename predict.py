import argparse
import os
import torch
import yaml
import numpy as np
import torch.nn.functional as F

import config_folder as cf
from data_loaders.Chairs import Chairs
from data_loaders.kitti import KITTI
from data_loaders.sintel import Sintel
from model import MaskFlownet, MaskFlownet_S, Upsample, EpeLossWithMask


def centralize(img1, img2):
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
    return img1 - rgb_mean, img2-rgb_mean, rgb_mean

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='model checkpoint to load')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='Batch Size')
parser.add_argument('-f', '--root_folder', type=str, default=None,
                    help='Root folder of KITTI')
parser.add_argument('--resize', type=str, default='')
args = parser.parse_args()
resize = (int(args.resize.split(',')[0]), int(args.resize.split(',')[1])) if args.resize else None
num_workers = 2

with open(os.path.join('config_folder', args.dataset_cfg)) as f:
    config = cf.Reader(yaml.load(f))
with open(os.path.join('config_folder', args.config)) as f:
    config_model = cf.Reader(yaml.load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = eval(config_model.value['network']['class'])(config)
checkpoint = torch.load(os.path.join('weights', args.checkpoint))

net.load_state_dict(checkpoint)
net = net.to(device)

if config.value['dataset'] == 'kitti':
    dataset = KITTI(args.root_folder, split='train', editions='mixed', resize=resize, parts='valid')
elif config.value['dataset'] == 'chairs':
    dataset = Chairs(args.root_folder, split='valid')
elif config.value['dataset'] == 'sintel':
    dataset = Sintel(args.root_folder, split='valid', subset='final')
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=False,
                                          batch_size=args.batch,
                                          num_workers=num_workers,
                                          drop_last=False,
                                          pin_memory=True)

epe = []
for idx, sample in enumerate(data_loader):
    with torch.no_grad():
        im0, im1, label, mask, path = sample
        if isinstance(mask, list):
            mask = torch.ones((label.shape[0], label.shape[1], label.shape[2], 1), dtype=label.dtype, device=device)

        im0 = im0.permute(0, 3, 1, 2)
        im1 = im1.permute(0, 3, 1, 2)
        im0, im1, _ = centralize(im0, im1)
        
        # The original MxNet implementation of MaskFlownet predict the flipped flow
        label = label.permute(0, 3, 1, 2).to(device).flip(1)
        
        mask = mask.permute(0, 3, 1, 2).to(device)

        shape = im0.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

        im0 = im0.to(device)
        im1 = im1.to(device)

        pred, flows, warpeds = net(im0, im1)

        up_flow = Upsample(pred[-1], 4)
        up_occ_mask = Upsample(flows[0], 4)

    if pad_h != 0 or pad_w != 0:
        up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
                  torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=device).view(1, 2, 1, 1)
        up_occ_mask = F.interpolate(up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')

    epe.append(EpeLossWithMask()(up_flow, label, mask).detach())
    
    # Flip the flow to get the final prediction
    final_flow = up_flow.flip(1)


print("AEPE: "+config.value['dataset'], torch.cat(epe).mean())
