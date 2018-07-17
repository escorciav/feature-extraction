import collections
import csv
import os
from pathlib import Path
from collections import OrderedDict, Sequence

import h5py
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


def create_image_csv(filename, dirname):
    "Dump CSV with frames from a list of videos"
    newfile = filename + '-img.csv'
    dirname = Path(dirname)
    with open(filename, 'r') as fr, open(newfile, 'w') as fw:
        fw.write('image,label\n')
        for line in fr:
            data = line.strip().split(',')
            video_name, label = data[0], 0
            if len(data) > 1:
                label = data[1]
            video_dir = dirname / video_name
            if not video_dir.is_dir():
                continue
            images = [os.path.join(video_name, i.name)
                      for i in video_dir.iterdir()]
            images.sort()
            for frame in images:
                fw.write(f'{frame},{label}\n')
    return newfile


def numpy_collate(batch):
    "Puts each data field into a numpy array with outer dimension batch size"
    elem_type = type(batch[0])
    if (elem_type.__module__ == 'numpy' and
        elem_type.__name__ != 'str_' and
        elem_type.__name__ != 'string_'):
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            return np.stack(batch)
        if elem.shape == ():
            # scalars
            raise NotImplementedError
    elif isinstance(batch[0], int):
        return np.array(batch, dtype=np.int)
    elif isinstance(batch[0], float):
        return np.array(batch, dtype=np.float32)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        raise NotImplementedError(f'WIP, call 911!\n'
                                  f'type: {type(batch)}, {type(batch[0])}')
    return


class AverageMeter(object):
    """Computes and stores the average and current value

    Credits:
        @pytorch team - Imagenet example
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DumpArray(object):
    "Serialized numpy-array as HDF5 when queue is filled"

    def __init__(self, dataset_name, queue_size=128, dirname='.', offset=0,
                 filename='{:06d}.hdf5'):
        self.dirname = Path(dirname)
        self.dataset_name = dataset_name
        self.filename = filename
        self.tick = offset
        self.queue = []
        self.max_size = queue_size

        if not self.dirname.exists():
            os.makedirs(self.dirname)

    def __call__(self, x):
        self.queue.append(x)
        if len(self.queue) == self.max_size:
            self.dump()
            self.queue = []

    def close(self):
        if len(self.queue) > 0:
            self.dump()

    def dump(self):
        filename = self.dirname / self.filename.format(self.tick)
        with h5py.File(filename, 'w') as fid:
            x = np.vstack(self.queue)
            fid.create_dataset(self.dataset_name, data=x,
                               chunks=True, compression='lzf')
        self.tick += 1


class ImageFromCSV(Dataset):
    """Load images from a CSV list.

    It is a replacement for ImageFolder when you are interested in a
    particular set of images. Indeed, the only different is the way the images
    and targets are gotten.

    Args:
        filename (str, optional): CSV file with list of images to read.
        root (str, optional) : files in filename are a relative path with
            respect to the dirname here. It reduces size of CSV but not in
            memory.
        fields (sequence, optional): sequence with field names associated for
            image paths and targets, respectively. If not provided, it uses
            the first two fields in the first row.
    """

    def __init__(self, filename, root='', fields=None, transform=None,
                 target_transform=None, loader=default_loader):
        self.root = Path(root)
        self.filename = filename
        self.fields = fields
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = self._make_dataset()

    def _make_dataset(self):
        with open(self.filename, 'r') as fid:
            reader = csv.DictReader(fid, fieldnames=self.fields)

            if self.fields is None:
                self.fields = reader.fieldnames
            else:
                check = [i in reader for i in self.fields]
                if not all(check):
                    raise ValueError(f'Missing fields in {self.filename}')

            imgs = []
            for i, row in enumerate(reader):
                img_name = row[self.fields[0]]
                path = self.root / img_name

                target = 0
                if len(self.fields) > 1:
                    target = row[self.fields[1]]

                imgs.append((path, target))
            return imgs

    def __getitem__(self, index):
        """Return item

        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target
                   class.
        """
        path, target = self.imgs[index]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.imgs)


class TrimModel(nn.Module):
    "Remove layers from pytorch model"

    def __init__(self, model, stop):
        super(TrimModel, self).__init__()
        assert isinstance(stop, int)
        self.blocks_of_interest = nn.Sequential(
            OrderedDict(list(model.named_children())[:stop]))

    def forward(self, input):
        x = self.blocks_of_interest(input)
        return x


class TrimVGGModel(nn.Module):
    """Remove layers from VGG-style model

    Note:
        This is a workaround for VGG models from torchvision/pytorch-zoo
    """

    def __init__(self, model, stop):
        super(TrimVGGModel, self).__init__()
        assert isinstance(stop, Sequence)
        if len(stop) != 2:
            raise ValueError(f'Incorrect value for Trimming {model_name}')
        if stop[0] == 0:
            self._only_conv_vgg(model, stop)
        else:
            self._chop_later_blocks(model, stop)

    def forward(self, input):
        x = self.blocks_of_interest(input)
        if self.second_part is not None:
            # the hack is below :see_no_evil:
            x = self.second_part(x.view(x.size(0), -1))
        return x

    def _only_conv_vgg(self, model, stop):
        "retain frist block and chop it up to stop[1]"
        assert stop[0] == 0
        block_0 = list(model.named_children())[stop[0]][1]
        self.blocks_of_interest = nn.Sequential(
            OrderedDict(list(block_0.named_children())[:stop[1]]))
        self.second_part = None

    def _chop_later_blocks(self, model, stop):
        "retain all blocks before stop[0] and chop stop[0] up to stop[1]"
        blocks = list(model.named_children())
        self.blocks_of_interest = nn.Sequential(OrderedDict(blocks[:stop[0]]))
        self.second_part = nn.Sequential(OrderedDict(list(
            blocks[stop[0]][1].named_children())[:stop[1]]))