import csv
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(listfile, csvfile, frm_fmt='{:06d}.jpg', offset=8):
    if len(os.path.splitext(csvfile)[1]) < 1:
        csvfile += '.csv'
    fieldnames = ['image', 'target']

    with open(listfile, 'r') as fr, open(csvfile, 'w', newline='') as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for line in fr:
            fields = line.rstrip().split()
            frame_i = frm_fmt.format(int(fields[1]) + offset)
            dirname = os.path.basename(fields[0])
            filename = os.path.join(dirname, frame_i)
            target = fields[2]

            writer.writerow({'image': filename, 'target': target})

if __name__ == '__main__':
    parser = ArgumentParser(description='CSV list from C3D-list',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--listfile', required=True,
                        help='Path to list file used to extract C3D features.')
    parser.add_argument('-o', '--csvfile', required=True,
                        help='Path to CSV file used for feature extraction')
    args = parser.parse_args()

    main(**vars(args))
