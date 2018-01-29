import matplotlib.pyplot as plt


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