import urllib.request
import zipfile
import os
import tensorflow as tf
import skimage
import numpy as np
import csv


def getImagesLabelsTrainReady():
    train_data_dir = os.path.join("data", "GTSRB", "Final_Training_Images")
    test_data_dir = os.path.join("data", "GTSRB", "Final_Test", "Images")
    test_data_class_file = os.path.join('data', 'GT-final_test.csv')

    train_images, train_labels = load_data(train_data_dir, test_data_class_file)
    test_images, test_labels = load_data(test_data_dir, test_data_class_file)

    img_size = 28
    num_rgb_channels = 3
    img_size_flat = img_size * img_size * num_rgb_channels
    img_shape = (img_size, img_size, num_rgb_channels)
    #@todo auto count implementieren
    num_classes = 43


def create_label_dict_from_csv_file(class_file_dir, test_data_dir):
    test_label_dict = {}
    with open(class_file_dir) as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=';')
        next(reader, None)

        for row in reader:
            test_label_dict[test_data_dir + '/' + row[0]] = row[7]

    return test_label_dict


def get_label_id(file_path, test_label_dict):
    return int(test_label_dict[file_path])


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


def load_data(data_dir, label_csv_file_dir):
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(directories)
    labels = []
    images = []
    if not directories:
        label_dir = os.path.join(data_dir)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]

        test_labels_ids = create_label_dict_from_csv_file(label_csv_file_dir, data_dir)

        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(get_label_id(f, test_labels_ids))
    else:

        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith(".ppm")]

            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))

    return images, labels


def resize_images(resize_images, image_size=(32, 32)):
    ''' resize all images given image size
        default value -> (32,32)
        you can change also rgb value '''

    print('resize images')
    return [skimage.transform.resize(image, image_size, mode='constant') for image in resize_images]


def store_tmp_images(images1, labels1, target_dir, file_names1):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print('filename size: ', len(file_names1))
    print('image size: ', len(images1))

    for i, image in enumerate(images1):
        path = os.path.join(target_dir, file_names1[i])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        skimage.imsave(os.path.join(target_dir, file_names1[i]), image)
