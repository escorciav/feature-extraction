import os

from PIL import Image
import numpy
import h5py

from y8m_model import YouTube8MFeatureExtractor

# Instantiate extractor. Slow if called first time on your machine, as it
# needs to download 100 MB.
extractor = YouTube8MFeatureExtractor()

path = ('/home/escorciav/datasets/jhmdb/interim/Frames/'
        'brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/')
images_name = ['00001.png', '00030.png']

test = 'tmp/000000.hdf5'
if os.path.isfile(test):
    with h5py.File(test, 'r') as fid:
        gt = fid['y8m_pool3'][:]
else:
    gt = None

for i, img_name in enumerate(images_name):
    image_file = os.path.join(path, img_name)
    im = numpy.array(Image.open(image_file))
    features = extractor(im)

    if gt is not None:
        numpy.testing.assert_almost_equal(features, gt[i, :], decimal=7)