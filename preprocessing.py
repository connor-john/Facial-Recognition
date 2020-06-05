import numpy as np
from keras.preprocessing import image
from glob import glob
from collections import Counter

# get data
def get_data(datapath):
    return glob(datapath + str('*'))

# load an image by height x width
def load_img(filepath, H, W):
    img = image.img_to_array(image.load_img(filepath, target_size = [H, W])).astype('uint8')
    return img

# greyscale of image
def to_grayscale(img):
    return img.mean(axis=-1)

# Images as arrays
def image_array(files, N, H, W):

    shape = (N, H, W)
    images = np.zeros(shape)
    for i, f in enumerate(files):
        img = to_grayscale(load_img(f, H, W)) / 255.
        images[i] = img
    
    return images

# Labels 
# file format like 'subject1.sad'
def get_labels(files, N):

    labels = np.zeros(N)

    for i, f in enumerate(files):
        filename = f.rsplit('/', 1)[-1]
        subject_num = filename.split('.', 1)[0]

        # filenames start from 1
        idx = int(subject_num.replace('subject', '')) - 1
        labels[i] = idx
    
    label_count = Counter(labels)

    return label_count

# prepare data
def prep_data(datapath, H, W):

    files = get_data(datapath)
    # img = load_img(filepath, H, W)
    N = len(files)
    images = image_array(files, N, H, W)
    label_count = get_labels(files, N)


