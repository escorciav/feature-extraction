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


def main(dirname, csvfile, filename, batch_size=512, dataset_name='feature'):
    compression_flags = dict(compression="gzip", compression_opts=9)
    # Parse csvfile
    video_series = pd.read_csv(csvfile)['image'].apply(
        lambda x: os.path.basename(os.path.dirname(x)))
    hdf5_buckets = sorted(glob.glob(os.path.join(dirname, '*.hdf5')))

    with h5py.File(filename, 'w') as fw:
        for video_id in video_series.unique():
            # Get buckets where data of ith video was stored
            video_indeces = video_series.index[video_series == video_id].values
            buckets = make_buckets(video_indeces, batch_size)
            feature_list = []
            for b_id, bucket in buckets:
                hdf5_bucket_i = hdf5_buckets[b_id]
                # Map indeces into hdf5 indeces based on batch_size
                indeces = np.remainder(bucket, batch_size)
                with h5py.File(hdf5_bucket_i, 'r') as fr:
                    feature_list.append(fr[dataset_name][indeces, ...])

            feature = np.vstack(feature_list)
            group = fw.create_group(video_id)
            group.create_dataset(dataset_name, data=feature, chunks=True,
                                 **compression_flags)


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
    parser.add_argument('-h5dn', '--dataset-name', default='resnet152_avgpool',
                        help='Name for HDF5 dataset')
    args = parser.parse_args()

    main(**vars(args))
