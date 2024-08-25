import random

from mnist_dataloader import MnistDataloader
from show_images import show_images
from random import randrange
from neural_network import NeuralNetwork


def show_random_images():
    rand_nums = [randrange(0, 9999) for _ in range(15)]
    rand_images = [images[x] for x in rand_nums]
    rand_labels = [f'digit: {labels[x]}' for x in rand_nums]
    show_images(rand_images, rand_labels)


dataloader = MnistDataloader('dataset/t10k-labels.idx1-ubyte', 'dataset/t10k-images.idx3-ubyte')
images, labels = dataloader.images, dataloader.labels

nn = NeuralNetwork(0.1)
rand_num = randrange(0, 9999)
rand_image = images[rand_num]
rand_label = labels[rand_num]
bias, weights = nn.compute_gradients(rand_image, rand_label)
nn.update_parameters(bias, weights)
print(nn.predict(rand_image))

