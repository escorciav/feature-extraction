import logging
import glob
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import numpy as np
import pandas as pd


def make_buckets(indeces, batch_size):
    # Buckets are described in terms of its bucket-id and real-indeces
    # Note: It assumes that frames were extracted in order i.e. the features
    # were extracted respecting the temporal order
    buckets_id, first_occurence = np.unique(
        np.floor_divide(indeces, batch_size), return_index=True)
    buckets = []
    for i, b_id in enumerate(buckets_id):
        start = first_occurence[i]
        end = None
        if len(buckets_id) > i + 1:
            end = first_occurence[i + 1]
        buckets.append((b_id, indeces[start:end]))
    return buckets


def main(args):
    logging.info('Packing features in a single HDF5')
    logging.info(args)
    h5ds_kwargs = dict(chunks=True)
    h5ds_kwargs['compression'] = args.compression
    h5ds_kwargs['compression_opts'] = args.compression_rate

    video_series = pd.read_csv(args.csvfile)['image'].apply(
        lambda x: os.path.basename(os.path.dirname(x)))
    hdf5_buckets = sorted(glob.glob(os.path.join(args.dirname, '*.hdf5')))

    with h5py.File(args.filename, 'w') as fw:
        for video_id in video_series.unique():
            # Get buckets where data of ith video was stored
            video_indeces = video_series.index[video_series == video_id].values
            buckets = make_buckets(video_indeces, args.batch_size)
            feature_list = []
            for b_id, bucket in buckets:
                hdf5_bucket_i = hdf5_buckets[b_id]
                # Map indeces into hdf5 indeces based on batch_size
                indeces = np.remainder(bucket, args.batch_size)
                with h5py.File(hdf5_bucket_i, 'r') as fr:
                    feature_list.append(fr[args.dataset_name][indeces, ...])

            feature = np.vstack(feature_list)
            group = fw.create_group(video_id)
            group.create_dataset(args.dataset_name, data=feature,
                                 **h5ds_kwargs)
    logging.info('Successful execution')


if __name__ == '__main__':
    description = ('Pack features into a HDF5 with videos as groups and '
                   'features individual datasets in each group.')
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dirname', required=True,
                        help='Path to of folder with HDF5 features')
    parser.add_argument('-i', '--csvfile', required=True,
                        help='Path to CSV file used for feature extraction')
    parser.add_argument('-o', '--filename', required=True,
                        help='Name of HDF5 file to be generated')
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        help='mini-batch size used during extraction')
    parser.add_argument('-h5dn', '--dataset-name', default='resnet152',
                        help='Name for HDF5 dataset')
    parser.add_argument('-ca', '--compression', default='gzip')
    parser.add_argument('-cr', '--compression-rate', default=9, type=int)
    # logging
    parser.add_argument('-log', '--loglevel', default='INFO',
                        help='Logging level')
    args = parser.parse_args()

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        level=numeric_level)

    main(args)
