import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
# %matplotlib inline


class ImageClassificator:

    total_iterations = 0

    def __init__(
            self,
            img_size,
            img_size_flat,
            num_classes,
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_channels=3
    ):
        self.filter_size1 = 5
        self.num_filters1 = 16
        self.filter_size2 = 5
        self.num_filters2 = 36
        self.fc_size = 128

        self.test_labels = test_labels
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.train_images = train_images
        self.train_labels = train_labels

        self.test_images = test_images
        self.test_labels = test_labels

        self.train_batch_size = 64
        self.test_batch_size = 256

        self.x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
        self.x_image = tf.reshape(self.x, [-1, img_size, img_size, num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        self.test_true_classes = np.argmax(test_labels, axis=1)

        self.y_pred = None
        self.y_pred_cls = None
        self.cross_entropy = None
        self.cost = None
        self.optimizer = None
        self.correct_prediction = None
        self.accuracy = None
        self.session = None

    def __one_hot_encode(self, test_labels):
        # Get all test labels (one hot Encoding to int labels)
        self.test_true_classes = np.argmax(test_labels, axis=1)
        pass

    # weights function
    def __new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    # Bias Function
    def __new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def __new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True):
        # defined Tensorflow Api
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # create weights
        weights = self.__new_weights(shape=shape)

        # create biases
        biases = self.__new_biases(length=num_filters)

        # create layer
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        # add biases
        layer += biases
        # use pooling if desired
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        # use relu
        layer = tf.nn.relu(layer)

        return layer, weights

    def __flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    def __new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        weights = self.__new_weights(shape=[num_inputs, num_outputs])
        biases = self.__new_biases(length=num_outputs)

        layer = tf.matmul(input, weights) + biases

        # use relu if desired
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    def build_graph(self):
        # create first conv layer
        layer_conv1, weights_conv1 = \
            self.__new_conv_layer(input=self.x_image,
                           num_input_channels=self.num_channels,
                           filter_size=self.filter_size1,
                           num_filters=self.num_filters1,
                           use_pooling=True)
        # create second conv layer
        layer_conv2, weights_conv2 = \
            self.__new_conv_layer(input=layer_conv1,
                           num_input_channels=self.num_filters1,
                           filter_size=self.filter_size2,
                           num_filters=self.num_filters2,
                           use_pooling=True)
        # flatten layer
        layer_flat, num_features = self.__flatten_layer(layer_conv2)
        # first fully connected layer
        layer_fc1 = self.__new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=self.fc_size,
                                 use_relu=True)
        # create second fully connected layer
        layer_fc2 = self.__new_fc_layer(input=layer_fc1,
                                 num_inputs=self.fc_size,
                                 num_outputs=self.num_classes,
                                 use_relu=False)

        self.y_pred = tf.nn.softmax(layer_fc2)
        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def create_tf_session(self):
        self.session = tf.Session()
        self.session.run()

    def next_batch(self, num, data, labels):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = data[idx]
        labels_shuffle = labels[idx]
        return data_shuffle, labels_shuffle

    def optimize(self, num_iterations):
        '''
        train network given num_iterations
        '''
        global total_iterations
        start_time = time.time()
        for i in range(total_iterations,
                       total_iterations + num_iterations):
            x_batch, y_true_batch = self.next_batch(self.train_batch_size, self.train_images, self.train_labels)
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}
            self.session.run(self.optimizer, feed_dict=feed_dict_train)
            if i % 100 == 0:
                acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(i, acc))
        total_iterations += num_iterations
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def print_test_accuracy(self):
        num_test = len(self.test_images)
        cls_pred = np.zeros(shape=num_test, dtype=np.int)
        i = 0
        while i < num_test:
            j = min(i + self.test_batch_size, num_test)
            images = self.test_images[i:j, :]
            labels = self.test_labels[i:j, :]
            feed_dict = {self.x: images,
                         self.y_true: labels}
            cls_pred[i:j] = session.run(self.y_pred_cls, feed_dict=feed_dict)
            i = j
        cls_true = self.test_true_classes
        correct = (cls_true == cls_pred)
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))