import os

import torch
from torch.utils.data import Dataset

from data_loaders.chairs.flo import load as load_flow
from data_loaders.chairs.ppm import load as load_ppm


class Chairs(Dataset):
    def __init__(self, root_path, split='train'):
        super(Chairs, self).__init__()
        self.image_list = []
        with open(os.path.join(root_path, 'FlyingChairs_train_val.txt')) as fp:
            for i in range(1, 22873):
                state = fp.readline()[0]
                if state == '1' and split == 'train':
                    self.image_list.append(os.path.join(root_path, 'data', '%05d_img1.ppm' % i))
                elif state == '2' and split == 'valid':
                    self.image_list.append(os.path.join(root_path, 'data', '%05d_img1.ppm' % i))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        im0_path = self.image_list[idx]
        im1_path = im0_path.replace('_img1.ppm', '_img2.ppm')
        flow_path = im0_path.replace('_img1.ppm', '_flow.flo')
        img0 = load_ppm(im0_path)
        img1 = load_ppm(im1_path)
        flow = load_flow(flow_path)

        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()

        return img0, img1, flow, [], im0_path
