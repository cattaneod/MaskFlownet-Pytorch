import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

VALIDATE_INDICES = dict()
VALIDATE_INDICES['2012'] = [0, 12, 15, 16, 17, 18, 24, 30, 38, 39, 42, 50, 54, 59, 60, 61, 77, 78, 81, 89, 97, 101, 107, 121, 124, 142, 145, 146, 152, 154, 155, 158, 159, 160, 164, 182, 183, 184, 190]
VALIDATE_INDICES['2015'] = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]


class KITTI(Dataset):
    def __init__(self, kitti_root, split='train', editions='mixed', parts='mixed', crop=None, resize=None, samples = None):
        self.kitti_root = kitti_root
        self.crop = crop
        self.resize = resize
        self.editions = editions
        self.split = split

        kitti_2012_image = os.path.join(kitti_root, r'kitti_stereo_2012/training/colored_0')
        kitti_2012_flow_occ = os.path.join(kitti_root, r'kitti_stereo_2012/training/flow_occ')
        kitti_2015_image = os.path.join(kitti_root, r'kitti_stereo_2015/training/image_2')
        kitti_2015_flow_occ = os.path.join(kitti_root, r'kitti_stereo_2015/training/flow_occ')
        kitti_path = dict()
        kitti_path['2012' + 'image'] = kitti_2012_image
        kitti_path['2012' + 'flow_occ'] = kitti_2012_flow_occ
        kitti_path['2015' + 'image'] = kitti_2015_image
        kitti_path['2015' + 'flow_occ'] = kitti_2015_flow_occ

        kitti_path['2012' + 'testing'] = os.path.join(kitti_root, r'kitti_stereo_2012/testing/colored_0')
        kitti_path['2015' + 'testing'] = os.path.join(kitti_root, r'kitti_stereo_2015/testing/image_2')
        editions = ('2012', '2015') if editions == 'mixed' else (editions, )

        self.image_list = []
        self.flow_list = []

        for edition in editions:
            if split == 'train':
                path_images = kitti_path[edition + 'image']
            else:
                path_images = kitti_path[edition + 'testing']
            path_flows = kitti_path[edition + 'flow_occ']
            num_files = len(os.listdir(path_flows)) - 1
            ind_valids = VALIDATE_INDICES[edition]
            num_valids = len(ind_valids)
            if samples is not None:
                num_files = min(num_files, samples)
            ind = 0
            for k in range(num_files):
                if split == 'train':
                    if ind < num_valids and ind_valids[ind] == k:
                        ind += 1
                        if parts == 'train':
                            continue
                    elif parts == 'valid':
                        continue
                    self.flow_list.append(os.path.join(path_flows, '%06d_10.png' % k))
                self.image_list.append(os.path.join(path_images, '%06d_10.png' % k))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        im0_path = self.image_list[idx]
        im1_path = im0_path.replace('_10.png', '_11.png')

        img0 = cv2.imread(im0_path)
        img1 = cv2.imread(im1_path)
        flow = []
        occ = []

        if self.split == 'train':
            flow_occ = cv2.imread(self.flow_list[idx], -1)

        if self.crop is not None:
            img0 = img0[-self.crop[0]:, :self.crop[1]]
            img1 = img1[-self.crop[0]:, :self.crop[1]]
            if self.split == 'train':
                flow_occ = flow_occ[-self.crop[0]:, :self.crop[1]]

        if self.split == 'train':
            flow = np.flip(flow_occ[..., 1:3], axis=-1).astype(np.float32)
            flow = (flow - 32768.) / (64.)
            occ = flow_occ[..., 0:1].astype(np.uint8)

        if self.resize is not None:
            img0 = cv2.resize(img0, self.resize)
            img1 = cv2.resize(img1, self.resize)
            if self.split == 'train':
                flow = cv2.resize(flow, self.resize) * ((np.array(self.resize, dtype = np.float32) - 1.0) / (
                        np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0))[np.newaxis, np.newaxis, :]
                occ = cv2.resize(occ.astype(np.float32), self.resize)[..., np.newaxis]
                flow = flow / (occ + (occ == 0))
                # occ = (occ * 255).astype(np.uint8)
        # elif self.split == 'train':
        #     occ = occ * 255

        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()

        return img0, img1, flow, occ, im0_path


if __name__ == '__main__':
    dataset = KITTI('/media/RAIDONE/DATASETS/KITTI/flow/', resize=(1152, 512))
    it = dataset.__getitem__(0)
    print(it)