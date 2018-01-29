import urllib.request
import zipfile
import os
import skimage
import numpy as np
import csv


class ImageLoader:

    def __init__(self):
        self.train_data_dir = os.path.join("data", "GTSRB", "Final_Training", "Images")
        self.test_data_dir = os.path.join("data", "GTSRB", "Final_Test", "Images")
        self.test_data_class_file = os.path.join('data', 'GT-final_test.csv')

        self.img_size = 28
        self.num_rgb_channels = 3
        self.img_size_flat = self.img_size * self.img_size * self.num_rgb_channels
        self.img_shape = (self.img_size, self.img_size, self.num_rgb_channels)
        self.num_classes = self.__count_classes(self.train_data_dir)

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def prepare_images_for_training(self):
        train_data_dir = os.path.join("data", "GTSRB", "Final_Training", "Images")
        test_data_dir = os.path.join("data", "GTSRB", "Final_Test", "Images")
        test_data_class_file = os.path.join('data', 'GT-final_test.csv')

        train_images, train_labels = self.__load_data(train_data_dir)
        test_images, test_labels = self.__load_data(test_data_dir, test_data_class_file)

        train_images = self.__resize_images(train_images, self.img_shape)
        test_images = self.__resize_images(test_images, self.img_shape)

        train_images = np.array(train_images)
        test_images = np.array(test_images)
        self.test_labels = np.array(test_labels)
        self.train_labels = np.array(train_labels)

        self.train_images = self.__flatten_images(train_images, self.img_size_flat)
        self.test_images = self.__flatten_images(test_images, self.img_size_flat)

    @staticmethod
    def __count_classes(data_dir):
        directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return len(directories)

    @staticmethod
    def __encode_labels_as_one_hot_labels(self, labels, num_classes):
        targets = labels.reshape(-1)
        one_hot_labels = np.eye(num_classes)[targets]
        return one_hot_labels

    @staticmethod
    def __flatten_images(images, img_size_flat):
        return images.flatten().reshape(len(images), img_size_flat)

    @staticmethod
    def __create_label_dict_from_csv_file(self, class_file_dir, test_data_dir):
        test_label_dict = {}
        with open(class_file_dir) as fin:
            reader = csv.reader(fin, skipinitialspace=True, delimiter=';')
            next(reader, None)

            for row in reader:
                test_label_dict[test_data_dir + '/' + row[0]] = row[7]

        return test_label_dict

    @staticmethod
    def __get_label_id(file_path, test_label_dict):
        return int(test_label_dict[file_path])

    @staticmethod
    def __download_unzip_data_set(target_dir, url):
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

    def __load_data(self, data_dir, label_csv_file_dir=None):
        labels = []
        images = []
        if not label_csv_file_dir:
            label_dir = os.path.join(data_dir)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith(".ppm")]

            test_labels_ids = self.__create_label_dict_from_csv_file(label_csv_file_dir, data_dir)

            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(self.__get_label_id(f, test_labels_ids))
        else:

            directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            for d in directories:
                label_dir = os.path.join(data_dir, d)
                file_names = [os.path.join(label_dir, f)
                              for f in os.listdir(label_dir) if f.endswith(".ppm")]

                for f in file_names:
                    images.append(skimage.data.imread(f))
                    labels.append(int(d))

        return images, labels

    @staticmethod
    def __resize_images(resize_images, image_size=(32, 32)):
        ''' resize all images given image size
            default value -> (32,32)
            you can change also rgb value '''
        print('resize images')
        return [skimage.transform.resize(image, image_size, mode='constant') for image in resize_images]

    @staticmethod
    def __store_tmp_images(images1, labels1, target_dir, file_names1):
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        print('filename size: ', len(file_names1))
        print('image size: ', len(images1))

        for i, image in enumerate(images1):
            path = os.path.join(target_dir, file_names1[i])
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            skimage.imsave(os.path.join(target_dir, file_names1[i]), image)
