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


shape = (28, 28)
dataloader = MnistDataloader('dataset/t10k-labels.idx1-ubyte', 'dataset/t10k-images.idx3-ubyte',
                             shape)
images, labels = dataloader.images, dataloader.labels

nn = NeuralNetwork(0.1, shape)
rand_num = randrange(0, 9999)
rand_image = images[rand_num]
rand_label = labels[rand_num]
for i in range(len(images)):
    bias, weights = nn.compute_gradients(images[i], labels[i])
    nn.update_parameters(bias, weights)
    if i % 100 == 0:
        print(f'Predicting for : {rand_label}')
        print(nn.predict(rand_image[0][0]))

