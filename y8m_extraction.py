import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from y8m_model import YouTube8MFeatureExtractor as Inception

from utils import AverageMeter, DumpArray, ImageFromCSV, numpy_collate

# cudnn.benchmark = True
# PIN_MEMORY = True


def main(args):
    logging.info('Y8M Inception extraction begins')
    logging.info('Loading model')
    net = Inception()

    # TODO add this to model, making transform method public
    logging.info('Setup transform model')
    img_transform = lambda x: np.array(x)

    logging.info('Setup dataloader')
    img_loader = DataLoader(
        dataset=ImageFromCSV(args.filename, args.root,
                             transform=img_transform),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=numpy_collate)  # ,
        # pin_memory=PIN_MEMORY)

    logging.info('Setup donkey for dumping')
    dump_helper = DumpArray(args.dataset_name, dirname=args.prefix)
    if args.print_freq < 1:
        args.print_freq *= len(img_loader)
    args.print_freq = max(1, int(args.print_freq))

    # TODO: add multimeter
    batch_time = AverageMeter()
    in_time = AverageMeter()
    out_time = AverageMeter()
    logging.info('Dumping features for: {} images'.format(len(img_loader)))
    end = time.time()
    for i, (img, _) in enumerate(img_loader):
        if i == args.warm_up:
            logging.info('Warm-up period is over')
            batch_time.reset()
            in_time.reset()
            out_time.reset()

        in_time.update(time.time() - end)

        end = time.time()
        output = net(img[0])
        batch_time.update(time.time() - end)

        end = time.time()
        dump_helper(output)
        out_time.update(time.time() - end)

        if (i + 1) % args.print_freq == 0:
            logging.info(
                f'[{i + 1}/{len(img_loader)}]\t'
                f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data-in {in_time.val:.3f} ({in_time.avg:.3f})\t'
                f'Data-out {out_time.val:.3f} ({out_time.avg:.3f})\t')
        end = time.time()
    dump_helper.close()

    logging.info(f'Batch {batch_time.avg:.3f}\t'
                 f'Data-in {in_time.avg:.3f}\t'
                 f'Data-out {out_time.avg:.3f}\t')
    logging.info('Successful execution')


if __name__ == '__main__':
    parser = ArgumentParser(description='Y8M Inception feature extraction',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True,
                        help='path to dataset')
    parser.add_argument('-f', '--filename', required=True,
                        help='CSV file with list of images')
    parser.add_argument('-o', '--prefix', required=True,
                        help='Prefix, including path, for output files.')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='mini-batch size')
    parser.add_argument('-q', '--queue-size', default=1024, type=int,
                        help='queue size for serializing data')
    parser.add_argument('-h5dn', '--dataset-name', default='y8m_pool3',
                        help='Name for HDF5 dataset')
    # Logging and verbosity
    parser.add_argument('--print-freq', '-p', default=0.1, type=float,
                        help='print frequency')
    parser.add_argument('--warm-up', '-lwu', default=24, type=int,
                        help='Number of warm-up mini-batches')
    parser.add_argument('-log', '--loglevel', default='INFO',
                        help='Logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(
        format=('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] '
                '%(message)s'),
        level=numeric_level)

    main(args)
