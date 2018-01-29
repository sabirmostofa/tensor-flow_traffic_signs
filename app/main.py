import os

from app.image_loader import ImageLoader

train_data_dir = os.path.join("data", "GTSRB", "Final_Training", "Images")
test_data_dir = os.path.join("data", "GTSRB", "Final_Test", "Images")
test_data_class_file = os.path.join('data', 'GT-final_test.csv')

img_size = 28

image_loader = ImageLoader(img_size, train_data_dir, test_data_dir, test_data_class_file)
image_loader.prepare_images_for_training()

