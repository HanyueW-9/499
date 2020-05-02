from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import gzip
import codecs
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import numpy
from scipy import ndimage

from six.moves import urllib

# Create model of CNN with slim api
def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 16, 16, 1])

        # For slim.conv2d, default argument values are like
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.conv2d(x, 32, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.dropout(net, is_training=is_training, scope='dropout2', keep_prob=0.75)  # 0.5 by default
        net = slim.flatten(net, scope='flatten3')

        # For slim.fully_connected, default argument values are like
        # activation_fn = nn.relu,
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.fully_connected(net, 128, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

# Create model of FCN with slim api
def FCN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 16, 16, 1])
        net = slim.flatten(x, scope='flatten')
        net = slim.fully_connected(net, 270, scope='fc1')
       
        net = slim.fully_connected(net, 270, scope='fc2')
       
        net = slim.fully_connected(net, 128, scope='fc3')
        
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

SOURCE_URL = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz'
DATA_DIRECTORY = "data"

# Params
IMAGE_SIZE = 16
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

# Download data
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def get_data(path):
    df = pd.read_table(path, header=None, sep = ' +', engine='python')
    datas = df.drop(0, axis=1).values
    labels = df[0].apply(lambda x: int(x)).values

# Extract the labels
def extract_labels(labels):
    #Extract the labels into a vector of int64 label IDs.
    labels = numpy.reshape(labels, [-1])
    one_hot_encoding = numpy.eye(NUM_LABELS)[labels]
    return one_hot_encoding

# Augment training data
def expend_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x) # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

def split_data(data_file, train_file, test_file, rate=0.8):
    with codecs.open(data_file, 'r', 'utf8') as f, \
        codecs.open(train_file, 'w', 'utf8') as fw1, \
        codecs.open(test_file, 'w', 'utf8') as fw2:
        data = f.readlines()
        train_data = data[:int(len(data)*rate)]
        test_data = data[int(len(data)*rate):]
        fw1.write(''.join(train_data))
        fw2.write(''.join(test_data))

def split_train_data(train_file, K=5):
    df = pd.read_table(train_file, header=None, sep=' +', engine='python')

    K, SEED = K, 0
    skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)

    datas = df.drop(0, axis=1) #.values
    labels = df[0].apply(lambda x: int(x)) #.values
    for fold, (train_idx, valid_idx) in enumerate(skf.split(datas, labels)):
        fold_dir = os.path.join(os.path.dirname(train_file), 'fold_{}'.format(fold))#, 'train')
        os.makedirs(fold_dir, exist_ok=True)
        datas.loc[train_idx].to_csv(os.path.join(fold_dir, 'x_train'), index=False, header=None, sep=' ')
        labels.loc[train_idx].to_csv(os.path.join(fold_dir, 'y_train'), index=False, header=None, sep=' ')
        datas.loc[valid_idx].to_csv(os.path.join(fold_dir, 'x_test'), index=False, header=None, sep=' ')
        labels.loc[valid_idx].to_csv(os.path.join(fold_dir, 'y_test'), index=False, header=None, sep=' ')
        # x_train, y_train, x_test, y_test

def split_datas():
    # splitting full data into train/test for model evaluation.
    data_file = 'data/zip.train'
    train_file = 'data/train.data'
    test_file = 'data/test.data'
    split_data(data_file, train_file, test_file, rate=0.8)

    # splitting train data into subtrain/validation for selecting a good regularization hyper-parameter value.
    split_train_data(train_file, K=5)

def load_test_data(test_file):
    df = pd.read_table(test_file, header=None, sep=' +', engine='python')
    datas = df.drop(0, axis=1).values
    labels = df[0].apply(lambda x: int(x)).values
    labels = numpy.reshape(labels, [-1])
    return datas, labels

def load_data(file):
    df = pd.read_table(file, header=None, sep=' +', engine='python').values
    # df = numpy.reshape(df, [-1])
    return df

# Prepare zip data
def prepare_ZIP_data(Kth=0):
    #
    train_data_filename = os.path.join('data/fold_{}'.format(Kth), 'x_train')
    train_labels_filename = os.path.join('data/fold_{}'.format(Kth), 'y_train')
    valid_data_filename = os.path.join('data/fold_{}'.format(Kth), 'x_test')
    valid_labels_filename = os.path.join('data/fold_{}'.format(Kth), 'y_test')

    train_data = load_data(train_data_filename)
    train_labels = load_data(train_labels_filename)
    train_labels = extract_labels(train_labels)
    valid_data = load_data(valid_data_filename)
    valid_labels = load_data(valid_labels_filename)
    valid_labels = extract_labels(valid_labels)


    test_file = 'data/test.data'
    test_data, test_labels = load_test_data(test_file)
    test_labels = extract_labels(test_labels)

    # test_labels = train_labels[:VALIDATION_SIZE,:]
    # train_data = train_data[VALIDATION_SIZE:, :]
    # train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    # if use_data_augmentation:
    #     train_total_data = expend_training_data(train_data, train_labels)
    # else:
    train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, valid_data, valid_labels, test_data, test_labels

# Prepare ZIP data
def prepare_ZIP_data2():
    train_file = 'data/train.data'
    train_data, train_labels = load_test_data(train_file)
    train_labels = extract_labels(train_labels)

    test_file = 'data/test.data'
    test_data, test_labels = load_test_data(test_file)
    test_labels = extract_labels(test_labels)
    valid_data = test_data
    valid_labels = test_labels

    train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]

    return train_total_data, train_size, valid_data, valid_labels, test_data, test_labels

if __name__ == '__main__':
    split_datas()
    # prepare_MNIST_data(Kth=0)
