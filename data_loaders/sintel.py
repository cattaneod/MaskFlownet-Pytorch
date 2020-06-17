import os
import re
import struct

import numpy as np
import skimage

import torch
from torch.utils.data import Dataset


class Sintel(Dataset):
    def __init__(self, root_path, split='train', subset='final'):
        super(Sintel, self).__init__()
        self.split = split
        self.flow_loader = Flo(1024, 436)
        split_samples = np.loadtxt(os.path.join(root_path, 'sintel_train_val_split.txt'))
        file_list = {}
        file_list['train'] = []
        file_list['valid'] = []
        file_list['test'] = []
        file_list['train+valid'] = []
        pattern = re.compile(r'frame_(\d+).png')
        c = 0
        for part in ('training', 'test'):
            for seq in os.listdir(os.path.join(root_path, part, 'clean')):
                frames = os.listdir(os.path.join(root_path, part, subset, seq))
                frames = list(sorted(map(lambda s: int(pattern.match(s).group(1)),
                                         filter(lambda s: pattern.match(s), frames))))

                for i in frames[:-1]:
                    entry = [
                        os.path.join(root_path, part, subset, seq, 'frame_{:04d}.png'.format(i)),
                        os.path.join(root_path, part, subset, seq, 'frame_{:04d}.png'.format(i + 1))]
                    if part == 'test':
                        file_list['test'].append(entry)
                    else:
                        if split_samples[c] == 1:
                            file_list['train'].append(entry)
                        else:
                            file_list['valid'].append(entry)
                        file_list['train+valid'].append(entry)
                        c += 1
        self.dataset = file_list

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        im0_path, im1_path = self.dataset[self.split][idx]
        flow_path = im0_path.replace('/clean/', '/flow/').replace('/final/', '/flow/').replace('.png', '.flo')
        mask_path = im0_path.replace('/clean/', '/invalid/').replace('/final/', '/invalid/')
        img0 = skimage.io.imread(im0_path)
        img1 = skimage.io.imread(im1_path)

        flow = []
        mask = []

        if self.split != 'test':
            mask = skimage.io.imread(mask_path)
            mask = 255 - np.expand_dims(mask, -1)
            flow = self.flow_loader.load(flow_path)

        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()

        return img0, img1, flow, mask, im0_path


class Flo:
    def __init__(self, w, h):
        self.__floec1__ = float(202021.25)
        self.__floec2__ = int(w)
        self.__floec3__ = int(h)
        self.__floheader__ = struct.pack('fii', self.__floec1__, self.__floec2__, self.__floec3__)
        self.__floheaderlen__ = len(self.__floheader__)
        self.__flow__ = w
        self.__floh__ = h
        self.__floshape__ = [self.__floh__, self.__flow__, 2]

        if self.__floheader__[:4] != b'PIEH':
            raise Exception('Expect machine to be LE.')

    def load(self, file):
        with open(file, 'rb') as fp:
            if fp.read(self.__floheaderlen__) != self.__floheader__:
                raise Exception('Bad flow header: ' + file)
            result = np.ndarray(shape=self.__floshape__,
                                dtype=np.float32,
                                buffer=fp.read(),
                                order='C')
            return result

    def save(self, arr, fname):
        with open(fname, 'wb') as fp:
            fp.write(self.__floheader__)
            fp.write(arr.astype(np.float32).tobytes())
