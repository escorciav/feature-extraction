"Merge multiple HDF5 files"
import argparse
import glob
from pathlib import Path

import h5py


def main(args):
    print(args)
    with h5py.File(args.filename, 'x') as fw:
        for filename_i in args.files:
            with h5py.File(filename_i, 'r') as fr:
                for item_name, item in fr.items():
                    fr.copy(item, fw, name=item_name)


if __name__ == '__main__':
    description = 'Merge HDF5s'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filename', type=Path, required=True,
                        help='Output file with all the HDF5s merged')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs='+', type=Path,
                       help='List of files to merge')
    group.add_argument('--wildcard', help='glob valid expression')

    args = parser.parse_args()
    if len(args.files) == 0 and args.wildcard is not None:
        args.files = glob.glob(args.wildcard)
    main(args)
