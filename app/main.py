import urllib.request
import zipfile
import os
import tensorflow as tf
from skimage import data, transform
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np


def download_unzip_data_set(target_dir, url):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    path = os.path.join(target_dir, target_dir + '.zip')
    print('download', url)
    urllib.request.urlretrieve(url, path)

    print('extract')
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall('data')
    zip_ref.close()

    os.remove(path)


def load_data(data_dir):
    ''' load all training and test data given path '''

    print('load data')
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    labels_func = []
    images_func = []
    file_names_func = []
    file_names_all = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names_func = [os.path.join(label_dir, f)
            for f in os.listdir(label_dir) if f.endswith(".ppm")]

        for f in file_names_func:
            images_func.append(data.imread(f))
            labels_func.append(int(d))
            file_names_all.append(f)

    print("Number of labels(classes): ", len(set(labels_func)))
    print("Number of images in dataset: ", len(images_func))

    return images_func, labels_func, file_names_all


def display_images_and_labels(images, labels):
    print('plot data')
    ''' plot first image of all labels'''

    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for j, label in enumerate(unique_labels):
        print(j)
        image = images[labels.index(label)]
        plt.subplot(7, 8, i)  # 7 rows , 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


def resize_images(resize_images, image_size=(32, 32)):
    ''' resize all images given image size
        default value -> (32,32)
        you can change also rgb value '''

    print('resize images')
    return [transform.resize(image, image_size, mode='constant') for image in resize_images]


def store_tmp_images(images1, labels1, target_dir, file_names1):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print('filename size: ', len(file_names1))
    print('image size: ', len(images1))

    for i, image in enumerate(images1):
        path = os.path.join(target_dir, file_names1[i])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        imsave(os.path.join(target_dir, file_names1[i]), image)


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
