import csv
import logging
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class ImageFromCSV(ImageFolder):
    """Load images from a CSV list.

    It is a replacement for ImageFolder when you are interested in a particular
    set of images. Indeed, the only different is the way the images and targets
    are gotten.

    Args:
        filename (str, optional): CSV file with list of images to read.
        root (str, optional) : files in filename are a relative path with
            respect to the dirname here. It reduces size of CSV but not in
            memory.
        fields (sequence, optional): sequence with field names associated for
            image paths and targets, respectively. If not provided, it uses the
            first two fields in the first row.

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


class DumpArray(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __class__(self, filename, x):
        with h5py.File(filename, 'w') as fid:
            fid.create_dataset(self.dataset_name, data=x, chunks=True)


def main(root, filename, prefix, workers, batch_size, dataset_name,
         print_freq):
    logging.info('ResNet extraction begins')
    logging.info('Loading model')
    model = models.resnet152(pretrained=True)
    # Lazy way to extract features before classifier
    model.fc = nn.Linear(model.fc.in_features, model.fc.in_features,
                         bias=False)
    model.fc.weight.data = torch.eye(model.fc.in_features)
    logging.info('Model loaded successfully')

    # Use gpu + set inference mode
    model.cuda()
    logging.info('Model resides on GPU')
    model.eval()
    logging.info('Eval/Inference mode set')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    img_loader = DataLoader(
        dataset=ImageFromCSV(filename, root, transform=img_transform),
        batch_size=batch_size,
        num_workers=workers)

    dump_helper = DumpArray(dataset_name)
    filename_fmt = prefix + '-{:06d}.hdf5'
    batch_time = AverageMeter()
    datain_time = AverageMeter()
    dataout_time = AverageMeter()
    logging.info('Dumping features for: {} images'.format(len(img_loader)))
    end = time.time()
    for i, (img, _) in enumerate(img_loader):
        img_var = Variable(img, volatile=True)
        img_var = img_var.cuda()
        datain_time.update(time.time() - end)

        end = time.time()
        output = model(img_var)
        batch_time.update(time.time() - end)

        end = time.time()
        output_arr = output.cpu().data.numpy()
        filename_i = filename_fmt.format(i)
        dump_helper(filename_i, output_arr)
        dataout_time.update(time.time() - end)

        if (i + 1) % print_freq == 0:
            msg = ('{0}/{1}]\t'
                   'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data-in {datain_time.val:.3f} ({datain_time.avg:.3f})\t'
                   'Data-out {dataout_time.val:.3f} ({dataout_time.avg:.3f})\t'
                   )
            logging.info(
                msg.format(i + 1, len(img_loader), batch_time=batch_time,
                           datain_time=datain_time, dataout_time=dataout_time))
        end = time.time()
    logging.info('Successful execution')


if __name__ == '__main__':
    parser = ArgumentParser(description='PyTorch ResNet feature extraction',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True,
                        help='path to dataset')
    parser.add_argument('-f', '--filename', required=True,
                        help='CSV file with list of images')
    parser.add_argument('-o', '--prefix', required=True,
                        help='Prefix, including path, for output files.')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('-h5dn', '--dataset-name', default='resnet152_avgpool',
                        help='Name for HDF5 dataset')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('-log', '--loglevel', default='DEBUG',
                        help='Logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        level=numeric_level)
    delattr(args, 'loglevel')

    main(**vars(args))
