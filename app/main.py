import urllib.request
import zipfile
import os
import tensorflow as tf
from skimage import data, transform
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np

















def train_data(X_train, y_train):
    X = tf.placeholder(tf.float64, [None, 32, 32, 3])
    Y = tf.placeholder(tf.int64, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    X_flat = tf.contrib.layers.flatten(X)

    count_labels = len(set(y_train))

    print('label count: ', count_labels)

    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(X_flat, count_labels, tf.nn.relu)

    predicted_labels = tf.arg_max(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, Y), dtype='float'))

    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(201):
            _, loss_value, a = sess.run(
                [train, loss, accuracy],
                feed_dict={X: X_train, Y: y_train})
            if i % 1 == 0:
                print('Loss: ', loss_value)
                print('Accuracy: ', a)


#download_unzip_data_set('data', 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip')
#download_unzip_data_set('data', 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip')

# create train path
train_data_dir = os.path.join('data', 'GTSRB', 'Final_Training', 'Images')

# create test path
test_data_dir = os.path.join('data', 'GTSRB', 'Final_Test', 'Images')


images, labels, file_names = load_data(train_data_dir)
images = resize_images(images)
#
#store_tmp_images(images, labels, os.path.join('data', 'tmp'), file_names)


resized_images_dir = os.path.join('data', 'tmp', 'data', 'GTSRB', 'Final_Training', 'Images')

#images, labels, _ = load_data(resized_images_dir)

# display_images_and_labels(images, labels)

labels_a = np.array(labels)
images_a = np.array(images)
train_data(images_a, labels_a)
