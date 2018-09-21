import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import create_image_csv
from utils import AverageMeter, DumpArray, ImageFromCSV, TrimModel

torch.set_grad_enabled(False)
PIN_MEMORY = True


def main(args):
    logging.info('ResNet extraction begins')
    logging.info('Loading model')
    model = TrimModel(
        'resnet152', models.resnet152(pretrained=True), -2)

    # Use gpu + set inference mode
    logging.info('Shipping model to GPU')
    model.to('cuda:0')
    logging.info('Setup model in inference mode')
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        normalize])

    if args.is_video_list:
        args.filename = create_image_csv(args.filename, args.root)
    img_loader = DataLoader(
        dataset=ImageFromCSV(args.filename, args.root,
                             transform=img_transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY)

    dump_helper = DumpArray(args.dataset_name, queue_size=args.queue_size,
                            dirname=args.prefix)
    batch_time = AverageMeter()
    in_time = AverageMeter()
    out_time = AverageMeter()
    logging.info(f'Dumping features for: {len(img_loader.dataset)} images')
    with torch.set_grad_enabled(False):
        end = time.time()
        for i, (img, _) in enumerate(img_loader):
            img_d = img.to('cuda:0')
            in_time.update(time.time() - end)

            end = time.time()
            output_d = model(img_d)
            if args.adaptive_pool:
                output_d = F.adaptive_avg_pool2d(output_d, output_size=(1, 1))
            if args.squeeze:
                if args.batch_size == 1:
                    raise ValueError('WIP: edge case. Increase batch size.')
                output_d = output_d.squeeze()
            batch_time.update(time.time() - end)

            end = time.time()
            output_h = output_d.to('cpu')
            output_arr = output_h.detach().numpy()
            dump_helper(output_arr)
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
    parser = ArgumentParser(description='PyTorch ResNet feature extraction',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True,
                        help='path to dataset')
    parser.add_argument('-f', '--filename', required=True,
                        help='CSV file with list of images')
    parser.add_argument('-o', '--prefix', required=True,
                        help='Prefix, including path, for output files.')
    # loader
    parser.add_argument('-j', '--num_workers', default=8, type=int,
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size')
    # Outputs
    parser.add_argument('-q', '--queue-size', default=1, type=int,
                        help='queue size for serializing data')
    parser.add_argument('-h5dn', '--dataset-name', default='resnet152_avgpool',
                        help='Name for HDF5 dataset')
    # optional
    parser.add_argument('-if', '--is-video-list', action='store_true',
                        help='CSV contain video names')
    parser.add_argument('--adaptive-pool', action='store_true',
                        help='Force avg pool over spatial dimension')
    parser.add_argument('--squeeze', action='store_true',
                        help='Remove singleton dimension')
    # Logging and verbosity
    parser.add_argument('--print-freq', '-p', default=0.1, type=float,
                        help='print frequency')
    parser.add_argument('-log', '--loglevel', default='DEBUG',
                        help='Logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        level=numeric_level)

    main(args)
