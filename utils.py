import collections
import csv
import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


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
        self.root = root
        self.filename = filename
        self.fields = fields
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = self._make_dataset()

    def _make_dataset(self):
        with open(self.filename, 'r') as fid:
            data = csv.DictReader(fid, fieldnames=self.fields)

            if self.fields is None:
                self.fields = data.fieldnames
            else:
                for i in self.fields:
                    if i not in data:
                        raise ValueError('Missing {} field in {}'
                                         .format(i, self.filename))

            imgs = []
            for i, row in enumerate(data):
                img_name = row[self.fields[0]]
                path = os.path.join(self.root, img_name)

                target = 0
                if len(self.fields) > 1:
                    target = row[self.fields[1]]

                imgs.append((path, target))
            return imgs

    def __getitem__(self, index):
        """

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
