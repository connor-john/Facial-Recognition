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

    return labels, label_count

# create train/test split of data
def create_datasets(img, n_subjects, N, images, labels):
    
    # Train on 3 images per subject
    n_test = 3 * n_subjects
    n_train = N - n_test

    # initialise datasets
    train_images = np.zeros([n_train] + list(img.shape))
    train_labels = np.zeros([n_train])

    test_images = np.zeros([n_test] + list(img.shape))
    test_labels = np.zeros([n_test])

    # populate the datasets
    count = {}
    train_i = 0
    test_i = 0

    for img, label in zip(images, labels):
    
        count[label] = count.get(label, 0) + 1

        if count[label] > 3:

            train_images[train_i] = img
            train_labels[train_i] = label

            train_i += 1
        
        else:

            test_images[test_i] = img
            test_labels[test_i] = label

            test_i += 1
    
    return train_images, train_labels, test_images, test_labels

# Create dictionary of training data
# {Data label :  indexes of data}
def create_dictionary(train_labels, test_labels):
    
    train_label2idx = {}
    test_label2idx = {} 

    for i, label in enumerate(train_labels):
        if label not in train_label2idx:
            train_label2idx[label] = [i]
        else:
            train_label2idx[label].append(i)

    for i, label in enumerate(test_labels):
        if label not in test_label2idx:
            test_label2idx[label] = [i]
        else:
            test_label2idx[label].append(i)
    
    return train_label2idx, test_label2idx

# prepare data
def prepare_data(datapath, H, W):

    files = get_data(datapath)
    
    img = load_img(np.random.choice(files), H, W)
    N = len(files)
    images = image_array(files, N, H, W)

    labels, label_count = get_labels(files, N)
    unique_labels = set(label_count.keys())
    n_subjects = len(label_count)
    
    train_images, train_labels, test_images, test_labels = create_datasets(img, n_subjects, N, images, labels)

    train_label2idx, test_label2idx = create_dictionary(train_labels, test_labels)


